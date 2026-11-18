# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Codex installer and deferred lifecycle entry points (local macOS only)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import typer

from particles.api.cli import run
from particles.api.cli._progress import confirm, heartbeat_paused
from particles.api.cli.hook import hook_app
from particles.api.cli.init import init_app
from particles.config import get_config, reset_config, sqlite_file_path
from particles.egress import run_armed


@init_app.command("codex")
def init_codex_cmd(
    store: str | None = typer.Option(None, "--store", help="Memory store handle."),
    config: Path | None = typer.Option(None, "--config", help="Absolute Particles config to pin."),
    command: str | None = typer.Option(None, "--command", help="Absolute particles executable."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Inspect and validate without writes or prompts."
    ),
    remove: bool = typer.Option(
        False, "--remove", help="Remove only unchanged installer-owned resources."
    ),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable plan; never prompts."),
    recall_token_file: Path | None = typer.Option(
        None, "--recall-token-file", help="Absolute checked reader key file for an engine install."
    ),
    capture_token_file: Path | None = typer.Option(
        None, "--capture-token-file", help="Absolute checked Codex capture key file."
    ),
    mcp_token_file: Path | None = typer.Option(
        None, "--mcp-token-file", help="Absolute checked MCP agent key file."
    ),
) -> None:
    """Install Codex for a local store or the engine named by its pinned config."""
    from particles.api.cli._claude_code import absolutize_sqlite_dsn, pinned_default_dsn
    from particles.api.cli._codex_install import (
        apply_plan,
        plan_install,
        plan_needs_lock,
        set_profile,
    )
    from particles.api.cli._harness_pins import mcp_registration_env
    from particles.api.cli._hook_state import installer_lock_refusal
    from particles.api.cli._scheduled_credentials import (
        credential_binding,
        credential_status,
        environment_key,
        reused_credential_status,
    )
    from particles.api.cli.init import (
        _config_yaml_path,
        _default_command_base,
        _default_store_dsn,
        _ensure_store_db,
        _plan_store_enable,
        _resolve_store_handle,
        _store_dsn,
    )

    previous = os.environ.get("PARTICLES_CONFIG")
    previous_engine = os.environ.get("PARTICLES_ENGINE_BASE_URL")
    os.environ["PARTICLES_ENGINE_BASE_URL"] = ""
    if config is not None:
        os.environ["PARTICLES_CONFIG"] = str(config.expanduser().resolve())
    reset_config()
    try:
        if sys.platform != "darwin":
            raise ValueError(
                "Codex integration currently supports local stores on macOS; "
                "Linux lifecycle and scheduling have not been measured"
            )
        path = _config_yaml_path()
        flags = {"recall": recall_token_file, "capture": capture_token_file, "mcp": mcp_token_file}
        if engine := get_config().engine.base_url:
            if config is not None and (not config.is_absolute() or config != path):
                raise ValueError("Codex thin mode requires an absolute config pin without links.")
            _init_codex_thin(
                path,
                engine,
                store or "default",
                command or _default_command_base(),
                flags,
                dry_run=dry_run,
                remove=remove,
                json_output=json_output,
            )
            return
        if any(flag is not None for flag in flags.values()):
            raise ValueError("Codex engine token options require an engine-backed config.")
        handle = _resolve_store_handle(store)
        path, new_text = _plan_store_enable(handle)
        text = new_text if new_text is not None else path.read_text()
        cfg = get_config()
        dsn = _store_dsn(handle) or _default_store_dsn(handle)
        if handle != "default" and absolutize_sqlite_dsn(dsn) != dsn:
            raise ValueError(
                "Codex refuses a relative named-store database; configure an absolute DSN"
            )
        dsn = absolutize_sqlite_dsn(dsn)
        home = Path(cfg.codex.home_dir).expanduser().resolve()
        state = Path(cfg.codex.state_dir).expanduser().resolve()
        env = {
            "PARTICLES_CONFIG": str(path),
            "PARTICLES_ENGINE_BASE_URL": "",
            "CODEX_HOME": str(home),
            "PARTICLES_CODEX_STATE_DIR": str(state),
            "PARTICLES_PROJECT_IDENTITY_LOCK": str(
                Path(cfg.codex.identity_lock_path).expanduser().resolve()
            ),
        }
        if handle == "default":
            env["DATABASE_URL"] = dsn
        blob = Path(cfg.storage.blob_dir).expanduser()
        if not blob.is_absolute():
            database = sqlite_file_path(dsn)
            blob = (Path(database).parent if database else path.parent) / blob
        env["PARTICLES_BLOB_DIR"] = str(blob.resolve())
        route = credential_binding(handle, "diagnostic-only", dsn=dsn)
        if route is not None and route.provider == "anthropic":
            from urllib.parse import urlsplit

            endpoint = urlsplit(route.endpoint)
            if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
                raise ValueError(
                    "Scheduled Anthropic endpoint must contain no credentials/query/fragment"
                )
            env["ANTHROPIC_BASE_URL"] = route.endpoint
        # The shared predicate deliberately preserves an explicit absolute default
        # selected for the MCP read tools, as the Claude installer already does.
        mcp_env = mcp_registration_env(env, dsn)
        mcp_dsn = mcp_env.get("DATABASE_URL") or pinned_default_dsn(mcp_env)
        mcp_blob = Path(cfg.storage.blob_dir).expanduser()
        if not mcp_blob.is_absolute():
            mcp_database = sqlite_file_path(mcp_dsn)
            mcp_blob = (Path(mcp_database).parent if mcp_database else path.parent) / mcp_blob
        mcp_env["PARTICLES_BLOB_DIR"] = str(mcp_blob.resolve())
        plan = plan_install(
            home=home,
            state=state,
            config=path,
            config_text=text,
            store=handle,
            dsn=dsn,
            command=command or _default_command_base(),
            env=env,
            mcp_env=mcp_env,
            agents=Path.home() / "Library" / "LaunchAgents",
            remove=remove,
        )
        reused = plan.schedule is not None and plan.owned_job is None
        profile = cfg.consolidation.credential_profile
        if not remove and not reused:
            profile = profile or "codex"
            text = set_profile(text, profile)
            for change in plan.files:
                if change.path == path:
                    change.after = text.encode()
        binding = credential_binding(handle, profile, dsn=dsn)
        status = (
            reused_credential_status(plan.schedule)
            if reused
            else credential_status(binding, use_environment=False)
            if binding
            else "missing"
        )
        if reused and not remove:
            plan.notes.append(
                "Reused job definition and credential profile were not changed or acquired."
            )
            if status not in {"job environment", "Keychain", "keyless local route"}:
                plan.notes.append(
                    "learning blocked: verify/repair the reused job's own secret delivery."
                )
        if not remove and not reused and status in {"missing", "locked/denied"}:
            plan.notes.append(f"learning blocked: {status} extraction credential")
            if (
                not dry_run
                and not json_output
                and sys.stdin.isatty()
                and binding
                and confirm(
                    "Enrol the selected extraction key in macOS Keychain for this store?",
                    default=False,
                )
            ):
                from particles.keychain import enrol_generic_password

                with heartbeat_paused():
                    key = environment_key(binding) or typer.prompt(
                        "Extraction API key", hide_input=True
                    )
                enrol_generic_password(binding.service, "extraction", key)
                status = credential_status(binding, use_environment=False)
        # Asked in a dry run too, which never reaches the lock in apply_plan.
        lock_refusal = installer_lock_refusal() if plan_needs_lock(plan) else None
        if lock_refusal is not None:
            raise ValueError(f"the installer cannot take its lock: {lock_refusal}")
        report = plan.report()
        report["credential"] = status
        report["dry_run"] = dry_run
        if not dry_run:

            def initialize() -> None:
                saved = {key: os.environ.get(key) for key in env}
                os.environ.update(env)
                reset_config()
                try:
                    run(_ensure_store_db(handle))
                    from particles.api.cli._codex_runtime import initialize_empty_scope

                    if not run(initialize_empty_scope(handle)):
                        plan.notes.append(
                            "Project recall blocked: existing store needs an explicit "
                            "particles memory rescope; installation did not change its attribution."
                        )
                    from particles.api.cli._codex_runtime import record_learning_status

                    record_learning_status(handle, status)
                finally:
                    for key, value in saved.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                    reset_config()

            apply_plan(plan, initialize=initialize)
            report["applied"] = True
        if json_output:
            typer.echo(json.dumps(report, indent=2))
        else:
            typer.echo(
                "Codex "
                + (
                    "removal planned"
                    if remove and dry_run
                    else "removed"
                    if remove
                    else "configured, learning blocked: awaiting hook trust"
                )
            )
            for note in plan.notes:
                typer.echo(note)
    except (ValueError, OSError) as exc:
        if json_output:
            typer.echo(json.dumps({"adapter": "codex", "status": "conflict", "error": str(exc)}))
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        if previous is None:
            os.environ.pop("PARTICLES_CONFIG", None)
        else:
            os.environ["PARTICLES_CONFIG"] = previous
        if previous_engine is None:
            os.environ.pop("PARTICLES_ENGINE_BASE_URL", None)
        else:
            os.environ["PARTICLES_ENGINE_BASE_URL"] = previous_engine
        reset_config()


def _init_codex_thin(
    config: Path,
    engine: str,
    store: str,
    command: str,
    flags: dict[str, Path | None],
    *,
    dry_run: bool,
    remove: bool,
    json_output: bool,
) -> None:
    from particles.api.cli._codex_install import (
        FileChange,
        _manifest_bytes,
        apply_plan,
        plan_install,
        plan_needs_lock,
        read_file,
    )
    from particles.api.cli._codex_remote import (
        ROLES,
        capture_enabled,
        check_key,
        checked_config,
        checked_endpoint,
        read_client_key,
        token_path,
    )
    from particles.api.cli._hook_state import installer_lock_refusal
    from particles.api.client.http import EngineHttpError, EngineUnreachableError, HttpBackend
    from particles.secrets import EngineTokenFileError

    cfg = get_config()
    checked_config(config)
    url = checked_endpoint(engine)
    if not store.strip() or store != store.strip():
        raise ValueError("Codex engine store handle must be nonempty and unpadded.")
    home = Path(cfg.codex.home_dir).expanduser().resolve()
    state = Path(cfg.codex.state_dir).expanduser().resolve()
    capture = capture_enabled()
    if flags["capture"] is not None and not capture and not remove:
        raise ValueError("Capture key enrollment requires codex.harvest.allow_remote and enabled.")
    clients = ["recall", "mcp"] + (["capture"] if capture else [])
    targets = {client: token_path(state, client) for client in clients}
    env = {
        "PARTICLES_CONFIG": str(config),
        "PARTICLES_ENGINE_BASE_URL": "",
        "PARTICLES_ENGINE_TOKEN": "",
        "CODEX_HOME": str(home),
        "PARTICLES_CODEX_STATE_DIR": str(state),
        "PARTICLES_PROJECT_IDENTITY_LOCK": str(
            Path(cfg.codex.identity_lock_path).expanduser().resolve()
        ),
    }
    mcp_env = env | {"PARTICLES_ENGINE_TOKEN_FILE": str(targets["mcp"])}
    raw_config = read_file(config)
    if raw_config is None:
        raise ValueError("Codex thin mode requires an existing config pin.")
    plan = plan_install(
        home=home,
        state=state,
        config=config,
        config_text=raw_config.decode(),
        store=store,
        dsn="",
        command=command,
        env=env,
        mcp_env=mcp_env,
        agents=Path.home() / "Library" / "LaunchAgents",
        remove=remove,
        engine=url,
        tokens={client: str(path) for client, path in targets.items()},
    )
    if plan_needs_lock(plan) and (lock_refusal := installer_lock_refusal()) is not None:
        raise ValueError(f"the installer cannot take its lock: {lock_refusal}")
    keys: dict[str, str] = {}
    file_report: dict[str, Any] = {}
    if not remove:
        for client, target in targets.items():
            source = flags[client] or target
            if not source.is_absolute():
                raise ValueError(f"{client} token source must be absolute.")
            file_report[client] = {
                "path": str(target),
                "source": str(source),
                "role": ROLES[client].value,
                "state": "present",
            }
            try:
                keys[client] = read_client_key(source, client)
            except EngineTokenFileError as exc:
                if exc.check != "missing":
                    raise
                file_report[client]["state"] = "missing"
                if dry_run:
                    continue
                if flags[client] is not None or json_output or not sys.stdin.isatty():
                    raise ValueError(f"Missing {client} key. Pass --{client}-token-file.")
                with heartbeat_paused():
                    keys[client] = typer.prompt(
                        f"Codex {client} engine key", hide_input=True
                    ).strip()
                from particles.api._keys import role_from_prefix

                if role_from_prefix(keys[client]) is not ROLES[client]:
                    raise ValueError(f"{client} key must have role {ROLES[client].value}.")
        if capture:
            plan.notes.append("Capture catch-up can re-read retained active and archived rollouts.")
        plan.notes.append("Review and trust the installed hooks in the Codex hook UI.")
    report = plan.report()
    report["dry_run"] = dry_run
    report["token_files"] = file_report
    report["capture_enabled"] = capture and not remove
    report["mint"] = {
        client: [
            "particles",
            "engine",
            "keys",
            "create",
            "--role",
            ROLES[client].value,
            "--principal",
            {
                "recall": "codex-recall",
                "capture": "codex-hook",
                "mcp": cfg.mcp.write.asserter_identity,
            }[client],
            *(["--reserved-principal"] if client == "capture" else []),
        ]
        for client in file_report
    }

    if not dry_run:
        if not remove:

            async def probe() -> dict[str, str | None]:
                projects: dict[str, str | None] = {}
                for client, key in keys.items():
                    backend = HttpBackend(token=key, base_url=url)
                    projects[client] = await check_key(
                        backend,
                        client,
                        asserter=cfg.mcp.write.asserter_identity,
                    )
                await HttpBackend(token=keys["recall"], base_url=url).digest(store)
                return projects

            try:
                projects = run_armed(asyncio.wait_for(probe(), cfg.codex.hook_deadline_seconds))
            except EngineUnreachableError:
                raise ValueError("Codex engine is unreachable. No files were installed.") from None
            except EngineHttpError as exc:
                raise ValueError(
                    f"Codex engine refused enrollment (HTTP {exc.status_code})."
                ) from None
            except TimeoutError:
                raise ValueError("Codex credential verification timed out.") from None
            plan.manifest["key_projects"] = projects
            manifest_path = state / "installations" / f"{plan.store_id}.json"
            for change in plan.files:
                if change.path == manifest_path:
                    change.after = _manifest_bytes(plan.manifest)
            for client, key in keys.items():
                target = targets[client]
                before = read_file(target)
                if before is not None:
                    # Protect an existing destination even when the source was a separate file.
                    from particles.secrets import check_engine_token_file

                    check_engine_token_file(str(target))
                plan.files.insert(
                    0, FileChange(target, before, (key + "\n").encode(), sensitive=True)
                )
        apply_plan(plan)
        report["applied"] = True
    if json_output:
        typer.echo(json.dumps(report, indent=2))
    else:
        typer.echo(f"Codex thin client: {url}, store {store!r}")
        if dry_run:
            typer.echo(json.dumps(report, indent=2))
        else:
            for note in plan.notes:
                typer.echo(note)


def _validate_hook_destination(store: str, destination: str | None) -> None:
    if destination is not None:
        from particles.api.cli._codex_remote import installed_destination

        installed_destination(store, destination)
    elif get_config().engine.base_url:
        raise ValueError("Codex hooks need an engine install. Re-run particles init codex.")


@hook_app.command("codex-worker", hidden=True)
def codex_worker_cmd(
    store: str = typer.Option("default", "--store"),
    session: str = typer.Option(..., "--session"),
    cwd: str = typer.Option(..., "--cwd"),
    catch_up: bool = typer.Option(False, "--catch-up"),
    destination: str | None = typer.Option(None, "--destination"),
) -> None:
    """Run bounded, detached capture without calling a model."""
    import signal

    from particles.api.cli._codex_runtime import capture_worker

    def expired(_signum: int, _frame: Any) -> None:
        raise TimeoutError("codex_worker_deadline")

    previous_handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, get_config().codex.harvest.worker_deadline_seconds)
    try:
        _validate_hook_destination(store, destination)
        run_armed(capture_worker(store, session, cwd, catch_up))
    except Exception:  # fail closed; launcher hint survives a killed/failed worker
        typer.echo("particles codex: worker failed; retained hint/startup scan can retry", err=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@hook_app.command("codex-recall", hidden=True)
def codex_recall_cmd(
    store: str = typer.Option("default", "--store"),
    session: str = typer.Option(..., "--session"),
    cwd: str = typer.Option(..., "--cwd"),
    reason: str = typer.Option("startup", "--reason"),
    destination: str | None = typer.Option(None, "--destination"),
) -> None:
    """Bounded recall child; no exposure or use-feedback accounting."""
    from particles.api.cli._codex_runtime import recall

    try:
        _validate_hook_destination(store, destination)
        output = run_armed(
            asyncio.wait_for(
                recall(store, session, cwd, reason), get_config().codex.hook_deadline_seconds
            )
        )
        if output:
            typer.echo(output)
    except Exception:
        typer.echo("particles codex: recall omitted; capture remains independent", err=True)
        typer.echo(
            json.dumps(
                {
                    "systemMessage": "Particles status/recall unavailable. "
                    "Run particles hook doctor --harness codex for diagnostics; "
                    "delivery remains unknown."
                }
            )
        )


async def codex_doctor(store: str) -> tuple[bool, list[str]]:
    """Human diagnostic, no source payloads, secret values or hosted probe calls."""
    import shutil

    from particles.api.cli._codex_install import digest, read_file
    from particles.api.cli._codex_runtime import (
        HOME_MOVED,
        _hint_age_seconds,
        health_summary,
        home_binding,
        paths,
    )
    from particles.api.cli._codex_schedule import discover, run_command
    from particles.api.cli._hook_state import read_json
    from particles.api.cli._project_identity import recorded_identity_roots
    from particles.api.cli._scheduled_credentials import (
        credential_binding,
        credential_status,
        reused_credential_status,
    )
    from particles.corpus.store import list_pending_snapshots_for_catchup
    from particles.db import schema_state, session_scope
    from particles.url_safety import trusted_which

    lines: list[str] = []
    good = True
    cfg = get_config()
    if not cfg.engine.base_url:
        base = Path(cfg.codex.state_dir).expanduser().resolve()
        for path in (base / "installations").glob("*.json"):
            old = read_json(path, {})
            if old.get("engine") and old.get("store") == store:
                return False, ["engine pin missing; re-run particles init codex before using hooks"]
    home, state, store_id = paths(store)
    try:
        home_binding(home)
    except ValueError:
        lines.append(HOME_MOVED)
        good = False
    # Looked up on the trusted PATH only, and run by run_command with that PATH:
    # an empty or relative entry would find a `codex` the working directory
    # holds (security review 2026-11-17-1, RA2).
    executable = trusted_which("codex")
    if executable:
        version = run_command([executable, "--version"]).stdout.strip()
        measured = "unmeasured" if version != "codex-cli 0.165.0-alpha.1.2" else "measured"
        lines.append(f"codex: {executable}; {version} ({measured}); parser codex-conversation/1")
    else:
        lines.append("codex: executable missing")
        good = False
    manifest = read_json(state.parent.parent / "installations" / f"{store_id}.json", {})
    remote = bool(get_config().engine.base_url)
    mode = "engine" if remote else "local"
    lines.append(
        f"ownership: {'recorded' if manifest else 'missing'}; {mode} store {store_id[:12]}"
    )
    good = good and bool(manifest)
    lines.append(
        "hooks: awaiting/verify trust in Codex hook UI; Particles never changes trust state"
    )
    lines.append("delivery: unknown; no exposure rows, utility credit or rank-lift feedback")
    lines.append(health_summary(store))
    learning = read_json(state / "learning.json", {})
    if learning:
        lines.append(
            f"scheduled credential check: {learning.get('credential', 'unknown')}; "
            f"checked at {learning.get('checked_at', 'unknown')}"
        )
    if not remote:
        lines.append(f"capture enabled: {get_config().codex.harvest.enabled}")
        good = good and get_config().codex.harvest.enabled
    cfg = get_config()
    lines.append(
        f"observer policies: codex={cfg.codex.observer_scope}; "
        f"claude-code={cfg.claude_code.observer_scope}"
    )
    if cfg.codex.observer_scope != cfg.claude_code.observer_scope:
        lines.append("warning: harness observer policies differ")
    if manifest:
        lines.append(
            f"pins: config={manifest.get('config')}; home={manifest.get('home')}; "
            f"store={manifest.get('store')}; exposure=unknown"
        )
    disk_path = next((p for p in (state, *state.parents) if p.exists()), state)
    lines.append(
        f"capture storage: {shutil.disk_usage(disk_path).free} bytes free; "
        f"writable={os.access(disk_path, os.W_OK)}"
    )
    pending = list((state / "pending").glob("*.json"))
    reasons: dict[str, int] = {}
    oldest: float | None = None
    for hint in pending:
        value = read_json(hint, {})
        reason = value.get("failure", "pending")
        reasons[reason] = reasons.get(reason, 0) + 1
        age = _hint_age_seconds(value.get("first_attempt"))
        if age is not None:
            oldest = max(oldest or 0, age)
    lines.append(
        f"capture hints: {len(pending)}; oldest age seconds: "
        f"{int(oldest) if oldest is not None else 'unknown/none'}; "
        f"failures: {json.dumps(reasons)}"
    )
    lines.append(
        "retention: no minimum rollout retention; "
        "deleting before capture can lose turns without trace"
    )
    if remote:
        ok, engine_lines = await _codex_engine_doctor(store, home, state, manifest)
        return good and ok, lines + engine_lines

    reused_job = None
    try:
        jobs = discover(Path.home() / "Library" / "LaunchAgents", store_id)
        for job in jobs:
            lines.append(
                f"schedule: {job.identity}; enabled={job.enabled}; loaded={job.loaded}; "
                f"{'owned' if manifest.get('job') == str(job.path) else 'reused'}"
            )
            good = good and job.enabled and job.loaded
            if manifest.get("job") != str(job.path):
                reused_job = job
            if (
                manifest.get("job") == str(job.path)
                and job.path is not None
                and digest(read_file(job.path)) != manifest.get("job_hash")
            ):
                lines.append("schedule: owned definition changed")
                good = False
        if not jobs:
            lines.append("schedule: missing; learning blocked")
            good = False
    except ValueError as exc:
        lines.append(f"schedule: conflict: {exc}")
        good = False
    binding = credential_binding(store)
    credential = (
        reused_credential_status(reused_job)
        if reused_job
        else credential_status(binding, use_environment=False)
        if binding
        else "missing profile"
    )
    from urllib.parse import urlsplit, urlunsplit

    endpoint = urlsplit(binding.endpoint) if binding else None
    netloc = endpoint.hostname or "" if endpoint else ""
    if ":" in netloc:
        netloc = f"[{netloc}]"
    if endpoint and endpoint.port:
        netloc += f":{endpoint.port}"
    shown_endpoint = (
        urlunsplit((endpoint.scheme, netloc, endpoint.path, "", "")) if endpoint else "none"
    )
    lines.append(f"extraction endpoint: {shown_endpoint}")
    lines.append(
        f"credential: {credential}; "
        + (f"provider={binding.provider}" if binding else "no profile")
    )
    good = good and credential in {"Keychain", "keyless local route", "job environment"}
    schema = await schema_state(store)
    if schema is None or schema.pending:
        from particles.api.cli._schema import migrate_command

        lines.append("store: uninitialized/behind; migrate with " + migrate_command(store))
        good = False
    else:
        async with session_scope(store) as session:
            count = len(await list_pending_snapshots_for_catchup(session))
        from datetime import UTC, datetime

        from particles.store.observer_scope_join import lens_may_engage

        async with session_scope(store) as session:
            engaged = await lens_may_engage(session)
        if not engaged and get_config().codex.observer_scope == "project":
            lines.append("project recall blocked: run particles memory rescope explicitly")
            good = False
        from particles.operations.consolidation import (
            CONSOLIDATION_ACTOR,
            _event_completed_at,
            _event_successful,
            latest_run_event,
        )

        async with session_scope(store) as session:
            last = await latest_run_event(session, actor=CONSOLIDATION_ACTOR)
            successful = await latest_run_event(
                session, actor=CONSOLIDATION_ACTOR, successful_only=True
            )
        lines.append(f"pending corpus: {count}; observer={get_config().codex.observer_scope}")
        if last:
            lines.append(
                f"last consolidation: {last.occurred_at}; "
                f"failed passes={not _event_successful(last)}; "
                f"degraded={bool((last.payload or {}).get('semantic_degraded'))}"
            )
        else:
            lines.append("last consolidation: none; first learning run has not succeeded")
        if successful:
            age = (
                datetime.now(UTC)
                - (_event_completed_at(successful) or successful.occurred_at.replace(tzinfo=UTC))
            ).total_seconds() / 3600
            due = age >= get_config().consolidation.min_interval_hours
            lines.append(
                f"consolidation due: {due}; last successful: {_event_completed_at(successful)}"
            )
        else:
            lines.append("consolidation due: true; no successful learning run")
    projects = Path(get_config().codex.state_dir).expanduser() / "projects"
    for project in projects.iterdir() if projects.exists() else []:
        if project.is_dir() and len(recorded_identity_roots(project.name)) > 1:
            lines.append(
                f"project_key_collision: {project.name}; "
                "repositories share one scope across harnesses"
            )
    return good, lines


async def _codex_engine_doctor(
    store: str,
    home: Path,
    state: Path,
    manifest: dict[str, Any],
) -> tuple[bool, list[str]]:
    from datetime import UTC, datetime
    from time import monotonic

    from particles import __version__
    from particles.api.cli._codex_install import _merge_hooks, _merge_mcp, read_file
    from particles.api.cli._codex_remote import (
        CodexKeyRefused,
        capture_enabled,
        installed_backend,
        installed_destination,
        prove_installed_key,
    )
    from particles.api.cli._hook_state import read_json
    from particles.api.cli._project_identity import (
        claude_project_slug,
        find_repository_root,
        observer_for_policy,
        recorded_identity_roots,
    )
    from particles.api.client.http import EngineHttpError, EngineUnreachableError, HttpBackend
    from particles.api.daemon import DaemonStatus
    from particles.secrets import EngineTokenFileError

    url, _, _ = installed_destination(store)
    cfg = get_config()
    budget = cfg.codex.hook_deadline_seconds
    lines = [f"engine: {url}; store={store}; local schema and schedules are inapplicable"]
    good = True
    try:
        hooks = read_file(home / "hooks.json")
        mcp = read_file(home / "config.toml")
        if (
            hooks is None
            or mcp is None
            or _merge_hooks(hooks, manifest["hooks"], manifest["hooks"]) != hooks
        ):
            raise ValueError
        if mcp.decode().count(manifest["mcp"]) != 1:
            raise ValueError
        _merge_mcp(mcp, manifest["mcp"], None)
    except (ValueError, OSError, KeyError):
        good = False
        lines.append("pins: installed Codex hooks or MCP registration differ; re-run init codex")
    handoff = read_json(state / "handoff.json", {})
    if handoff.get("result") == "worker launched":
        lines.append(f"worker hand-off: launched; last age seconds={_hint_age_for_doctor(handoff)}")
    else:
        lines.append("worker hand-off: no recorded launch")
    active_capture = capture_enabled()
    enrolled_capture = "capture" in manifest.get("tokens", {})
    lines.append(
        f"capture: {'enabled' if active_capture and enrolled_capture else 'disabled'}; "
        f"opt-in={cfg.codex.harvest.allow_remote}; credential enrolled={enrolled_capture}"
    )
    if active_capture != enrolled_capture:
        good = False
        lines.append("capture pins differ from current consent; re-run init codex")

    started = monotonic()
    try:
        public = await asyncio.wait_for(HttpBackend(token="", base_url=url).probe_health(), budget)
        if public.status_code >= 400:
            return False, lines + [f"reachability: engine answered HTTP {public.status_code}"]
        health = public.json()
    except (EngineHttpError, TimeoutError, ValueError):
        return False, lines + ["reachability: engine unreachable or health response refused"]
    lines.append(f"reachability: engine answered in {monotonic() - started:.2f} s")
    version = health.get("version")
    good = good and version == __version__ and health.get("status") == "ok"
    lines.append(
        f"version: engine {version!r}; client {__version__}"
        + ("; mismatch" if version != __version__ else "")
    )
    daemon = health.get("daemon")
    for client in ["recall", "mcp"] + (["capture"] if enrolled_capture else []):
        try:
            backend, key_manifest = installed_backend(store, client)
            await asyncio.wait_for(prove_installed_key(backend, client, key_manifest), budget)
            lines.append(f"{client} key: authenticated")
            if client == "recall":
                if daemon is None:
                    reply = await asyncio.wait_for(backend.probe_health(), budget)
                    daemon = reply.json().get("daemon")
                root = find_repository_root(Path.cwd())
                project = (
                    observer_for_policy(claude_project_slug(root), cfg.codex.observer_scope)
                    if root
                    else None
                )
                if project and len(recorded_identity_roots(project)) > 1:
                    lines.append(f"project_key_collision: {project}")
                digest = await asyncio.wait_for(
                    backend.digest_located(store, project, strict_scope=project is not None), budget
                )
                if project and digest.observer_engaged is not True:
                    good = False
                    lines.append("recall: engine did not confirm the requested project scope")
                else:
                    lines.append(f"recall: store answered; observer={project or 'store probe'}")
        except EngineTokenFileError as exc:
            good = False
            lines.append(f"{client} key file: {exc.check}")
        except CodexKeyRefused as exc:
            good = False
            lines.append(str(exc))
        except EngineUnreachableError:
            good = False
            lines.append(f"{client} probe: engine stopped answering")
        except EngineHttpError as exc:
            good = False
            reason = {
                401: "unauthenticated, invalid or revoked credential",
                403: "role or namespace refused",
                404: "store or endpoint missing",
                409: "project scope unavailable",
            }.get(exc.status_code or 0, f"HTTP {exc.status_code}")
            lines.append(f"{client} probe: {reason}")
        except (TimeoutError, ValueError, OSError):
            good = False
            lines.append(f"{client} probe: pin, response or deadline check failed")

    try:
        tasks = DaemonStatus.model_validate(daemon) if daemon is not None else None
    except ValueError:
        tasks = None
    task = (
        next((item for item in tasks.tasks if item.name == "consolidation"), None)
        if tasks
        else None
    )
    if tasks is None or not tasks.enabled or task is None:
        good = False
        lines.append("consolidation: no engine daemon schedule reported")
    else:
        last = task.last_finished_at or task.last_started_at
        if last is not None and last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        stale = (
            last is not None
            and (datetime.now(UTC) - last).total_seconds() > 2 * task.interval_seconds
        )
        degraded = (
            not tasks.healthy
            or task.last_error is not None
            or task.state == "crashed"
            or (task.last_outcome or "").startswith("completed, failed pass(es):")
        )
        good = good and not stale and not degraded
        lines.append(
            f"consolidation: {task.state}; last={last or 'none'}; "
            f"stale={stale}; degraded={degraded}"
        )
    return good, lines


def _hint_age_for_doctor(handoff: dict[str, Any]) -> str:
    from particles.api.cli._codex_runtime import _hint_age_seconds

    age = _hint_age_seconds(handoff.get("at"))
    return str(int(age)) if age is not None else "unknown"


@hook_app.command("codex-dismiss")
def codex_dismiss_cmd(
    session: str | None = typer.Option(
        None, "--session", help="Canonical session UUID to dismiss."
    ),
    acknowledge: bool = typer.Option(
        False, "--acknowledge", help="Acknowledge the current UI failure signature; retain hints."
    ),
    store: str = typer.Option("default", "--store", help="Local store handle."),
) -> None:
    """Acknowledge an unresolved capture hint; this cannot recover lost testimony."""
    # Unresolved hints clear only after capture or an explicit dismissal.
    # This operator-only alternative never deletes corpus data or a rollout.
    from particles.api.cli._codex_runtime import paths
    from particles.api.cli._hook_state import advisory_lock, read_json
    from particles.extraction.codex_transcript import session_uuid

    try:
        if acknowledge == (session is not None):
            raise ValueError("choose exactly one of --session or --acknowledge")
        if acknowledge:
            from particles.api.cli._codex_runtime import acknowledge_warning

            acknowledge_warning(store)
            typer.echo(
                "Current failure warning acknowledged; hints/corpus retained. "
                "New failures warn again."
            )
            return
        identity = session_uuid(session)
        _home, state, _store_id = paths(store)
        with advisory_lock(state / "capture.lock") as acquired:
            if not acquired:
                raise ValueError("capture is running; retry dismissal later")
            hint = state / "pending" / f"{identity}.json"
            if read_json(hint) is None:
                raise ValueError("no unresolved hint for this session")
            hint.unlink()
        typer.echo("Capture hint acknowledged; corpus retained. Lost turns remain unavailable.")
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
