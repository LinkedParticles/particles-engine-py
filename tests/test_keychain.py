# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The SDK adapter never prompts on reads and always restores process state."""

import ctypes
from types import SimpleNamespace
from typing import Any

import pytest

from particles import keychain


@pytest.mark.parametrize("status", [-25300, -25308, -25293, 0])
def test_read_disables_ui_and_restores_it_on_every_outcome(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    changes: list[int] = []
    released: list[Any] = []
    raw = ctypes.create_string_buffer(b"dummy-credential")

    class Fake:
        cf = SimpleNamespace(CFRelease=released.append)

        def call(self, name: str, *args: Any) -> int:
            if name == "SecKeychainGetUserInteractionAllowed":
                args[0]._obj.value = 1
            elif name == "SecKeychainSetUserInteractionAllowed":
                changes.append(int(args[0]))
            elif name == "SecKeychainCopyDefault":
                args[0]._obj.value = 12
            elif name == "SecKeychainFindGenericPassword":
                assert changes == [0]
                if status == 0:
                    args[5]._obj.value = len(b"dummy-credential")
                    args[6]._obj.value = ctypes.addressof(raw)
                    args[7]._obj.value = 13
                return status
            return 0

        def check(self, name: str, *args: Any) -> None:
            result = self.call(name, *args)
            if result:
                raise keychain.KeychainError(result)

    monkeypatch.setattr(keychain, "_API", Fake)
    if status in {-25308, -25293}:
        with pytest.raises(keychain.KeychainError, match="locked/denied"):
            keychain.read_generic_password("dummy-service", "dummy-account")
    else:
        assert keychain.read_generic_password("dummy-service", "dummy-account") == (
            "dummy-credential" if status == 0 else None
        )
    assert changes == [0, 1]
    assert released


def test_unsupported_platform_does_not_load_frameworks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keychain.sys, "platform", "linux")
    with pytest.raises(keychain.KeychainError, match="unavailable"):
        keychain.read_generic_password("service", "account")
