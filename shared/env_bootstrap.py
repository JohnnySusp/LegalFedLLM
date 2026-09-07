from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_UNSAFE_SECRET_PREFIXES = (
    "development-",
    "placeholder-",
    "replace-this-",
)

_ROLE_GENERATED_SECRETS = {
    "development": (
        "ADMIN_TOKEN",
        "REGISTRATION_TOKEN",
        "INTERNAL_API_TOKEN",
        "CLIENT_ADMIN_TOKEN",
        "QWEN_CLIENT_ADMIN_TOKEN",
        "GRANITE_CLIENT_ADMIN_TOKEN",
    ),
    "host": (
        "ADMIN_TOKEN",
        "INTERNAL_API_TOKEN",
    ),
    "client": (
        "QWEN_CLIENT_ADMIN_TOKEN",
        "GRANITE_CLIENT_ADMIN_TOKEN",
    ),
}

_ROLE_REQUIRED_SUPPLIED = {
    "development": (),
    "host": (),
    "client": ("REGISTRATION_TOKEN",),
}


@dataclass(frozen=True, slots=True)
class EnvironmentBootstrapResult:
    path: Path
    role: str
    created: bool
    values: dict[str, str]


def parse_env_text(content: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        values[key] = value
    return values


def load_env_file(
    path: str | Path = ".env",
    *,
    override: bool = False,
) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.is_file():
        return {}
    values = parse_env_text(env_path.read_text(encoding="utf-8"))
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def _validate_role(role: str) -> str:
    normalized = role.strip().lower()
    if normalized not in _ROLE_GENERATED_SECRETS:
        raise ValueError("bootstrap role must be development, host or client")
    return normalized


def _validate_secret(name: str, value: str) -> str:
    stripped = value.strip()
    if not stripped or stripped.startswith(_UNSAFE_SECRET_PREFIXES):
        raise ValueError(f"{name} must contain a non-placeholder secret")
    return stripped


def _replace_assignment(line: str, value: str) -> str:
    key = line.split("=", 1)[0]
    return f"{key}={value}"


def _render_template(
    content: str,
    *,
    role: str,
    replacements: Mapping[str, str],
) -> tuple[str, dict[str, str]]:
    values = parse_env_text(content)
    if "LEGALFEDLLM_BOOTSTRAP_ROLE" not in values:
        raise ValueError("environment template is missing LEGALFEDLLM_BOOTSTRAP_ROLE")
    if values["LEGALFEDLLM_BOOTSTRAP_ROLE"].strip().lower() != role:
        raise ValueError("environment template bootstrap role does not match the request")

    effective = dict(replacements)
    for name in _ROLE_REQUIRED_SUPPLIED[role]:
        if name not in effective:
            raise ValueError(f"{name} must be supplied when bootstrapping a {role} role")
        effective[name] = _validate_secret(name, effective[name])

    for name in _ROLE_GENERATED_SECRETS[role]:
        if name not in values:
            raise ValueError(f"environment template is missing {name}")
        if name not in effective:
            effective[name] = secrets.token_urlsafe(32)

    rendered_lines: list[str] = []
    seen: set[str] = set()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            key = line.split("=", 1)[0].strip()
            if key in effective:
                rendered_lines.append(_replace_assignment(line, effective[key]))
                seen.add(key)
                continue
        rendered_lines.append(line)

    missing = sorted(set(effective) - seen)
    if missing:
        raise ValueError(
            "environment template is missing replacement fields: "
            + ", ".join(missing)
        )

    rendered = "\n".join(rendered_lines).rstrip() + "\n"
    rendered_values = parse_env_text(rendered)
    for name in _ROLE_GENERATED_SECRETS[role]:
        _validate_secret(name, rendered_values[name])
    for name in _ROLE_REQUIRED_SUPPLIED[role]:
        _validate_secret(name, rendered_values[name])
    return rendered, rendered_values


def _write_private_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{secrets.token_hex(8)}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def load_or_create_env(
    *,
    path: str | Path,
    template_path: str | Path,
    role: str,
    replacements: Mapping[str, str] | None = None,
) -> EnvironmentBootstrapResult:
    normalized_role = _validate_role(role)
    env_path = Path(path)
    template = Path(template_path)

    if env_path.exists():
        if not env_path.is_file():
            raise ValueError(f"environment path is not a file: {env_path}")
        values = parse_env_text(env_path.read_text(encoding="utf-8"))
        existing_role = values.get("LEGALFEDLLM_BOOTSTRAP_ROLE", "").strip().lower()
        if existing_role != normalized_role:
            raise ValueError(
                f"existing {env_path} belongs to bootstrap role "
                f"{existing_role or 'unknown'!r}, not {normalized_role!r}"
            )
        for name in _ROLE_GENERATED_SECRETS[normalized_role]:
            if name not in values:
                raise ValueError(f"existing {env_path} is missing {name}")
            _validate_secret(name, values[name])
        for name in _ROLE_REQUIRED_SUPPLIED[normalized_role]:
            if name not in values:
                raise ValueError(f"existing {env_path} is missing {name}")
            _validate_secret(name, values[name])
        return EnvironmentBootstrapResult(
            path=env_path,
            role=normalized_role,
            created=False,
            values=values,
        )

    if not template.is_file():
        raise ValueError(f"environment template does not exist: {template}")
    rendered, values = _render_template(
        template.read_text(encoding="utf-8"),
        role=normalized_role,
        replacements=replacements or {},
    )
    _write_private_atomic(env_path, rendered)
    return EnvironmentBootstrapResult(
        path=env_path,
        role=normalized_role,
        created=True,
        values=values,
    )
