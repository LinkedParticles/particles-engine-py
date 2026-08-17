# AGENTS.md

Loaded by agents when working in this repository.

## Committing

Commits should be made with `uv run git commit -s` rather than a bare
`git commit`. The pre-commit hook needs `pre-commit` on PATH, and it is already
in `.venv/bin`, so `uv run` puts it there for that one command. A bare
`git commit` fails with `pre-commit not found`.

`--no-gpg-sign` must never be passed. Commit signing is configured in git and
every commit on this project carries a signature.

## Running commands

Never prepend `export PATH="$PWD/.venv/bin:$PATH"` to a command. `uv run`
already executes inside the project venv, so the prefix is dead weight, and
that `export` line trips a shell-safety check that forces a permission prompt
on every call it rides on.

`git add -A` should be avoided when another working tree or a parallel session
may have unstaged changes. Stage specific paths instead.
