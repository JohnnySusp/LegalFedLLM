# LegalFedLLM

LegalFedLLM is a protocol-first proof of concept for **bidirectional federated
knowledge transfer between heterogeneous language models** in legal-domain
workloads.

The project does not aggregate LoRA tensors across incompatible models. Each
participant keeps its own base model and model-native LoRA adapter local. What
crosses the federation boundary is a signed **Knowledge Package** produced over
a common reference dataset: retained token IDs and logits, answer-token loss
evidence, model/tokenizer identities, dataset identities, hashes, signatures,
and related provenance.

The topology below shows the model combination used in the project's physical
experiments. The same models are also used throughout this README as examples
of a heterogeneous LegalFedLLM deployment:

```text
Windows Client                         Linux Client
Qwen3 1.7B + PEFT LoRA                Granite 3.3 2B + PEFT LoRA
private local data                     private local data
        │                                      │
        ├──── signed Client Knowledge Package ─┤
        │                                      │
        └──────────────────┬───────────────────┘
                           ↓
                     Coordinator
                           ↓
              verification + safety/trust
                           ↓
       Client-specific DTW tokenizer alignment
                           ↓
                  DualMinCE selection
                           ↓
             Mistral Nemo Host + PEFT LoRA
                           ↓
                  hidden D^V validation
                           ↓
                 promote or roll back
                           ↓
              signed Host Knowledge Package
                     ↙             ↘
          Client-owned reverse alignment/distillation
```

No Client LoRA tensor is sent to the Coordinator or Host. Raw private Client
training examples remain on the Client machine.

> **Research boundary:** LegalFedLLM is a thesis proof of concept, not a
> production federated-learning platform. The repository contains real-model
> training, heterogeneous multi-Client knowledge transfer, desktop Clients and
> Host/Coordinator orchestration, but it does not claim formal differential
> privacy, production-calibrated poisoning detection, production identity
> management, or encrypted transport/storage.

## Hardware Requirements

Hardware requirements depend on the selected Client and Host models, numerical
precision, sequence length, LoRA configuration, batch size, reference-dataset
size, and the amount of training or validation performed on the GPU. LegalFedLLM
therefore does not define one universal VRAM or RAM minimum for every possible
model profile.

The following is a practical baseline for the small-model desktop configuration
used during development and experimentation:

| Role/resource | Baseline requirement | Experimental example |
| --- | --- | --- |
| Client GPU | NVIDIA CUDA-capable GPU with enough VRAM for the selected model and LoRA workload. Around 6–8 GB VRAM is a useful starting range for the small Client profiles used here. | Qwen3 1.7B was exercised on a Windows Client with about 6 GB VRAM; Granite 3.3 2B was exercised on a Linux Client with about 8 GB VRAM. |
| Client system RAM | 16 GB or more is a sensible starting point for small Client models, especially because the constrained-GPU path can offload saved activations to system RAM. | The development configuration included a 16 GB RAM / 8 GB VRAM Linux Client. |
| Host GPU | The Host normally needs substantially more VRAM than a Client because it may use a larger model and performs its own LoRA training and validation. | The Mistral Nemo Host experiments used an NVIDIA A40 with roughly 48 GB VRAM. |
| Storage | Allow substantial free SSD space for Python environments, Docker images on Linux, Transformers snapshots, Ollama models, PEFT checkpoints, caches, logs, and runtime artifacts. Tens of gigabytes can be consumed even by a small-model setup. | Qwen3 1.7B, Granite 3.3 2B, Mistral Nemo, their caches, and the desktop runtimes were stored separately during testing. |
| CPU/network | A modern multi-core x86_64 CPU and a stable network connection are sufficient for orchestration; the real training/inference path is primarily GPU-bound. | Clients reached the Coordinator through SSH forwarding while model computation remained local to each participant. |

These figures are examples from the models used in the project, not fixed
requirements of the LegalFedLLM protocol. A smaller model may run with less
memory, while a larger model, longer sequences, different precision, or more
aggressive training settings may require substantially more VRAM, RAM, and
storage. The built-in Low VRAM Mode and admission checks reduce memory pressure
for constrained Clients but do not guarantee that an arbitrary model will fit a
particular GPU.

## Windows x64

> **Windows release status:** the native portable Windows Client is implemented
> as a thin `LegalFedLLM.exe` launcher with the application source/runtime beside
> it. The Windows desktop path has been exercised on real x64 hardware with the
> Qwen3 1.7B Client in the heterogeneous experimental topology described above.

The Windows release requires installed Python 3.14 x64 and bootstraps or reuses
a release-local `.venv`. Persistent LegalFedLLM state remains in the sibling
`LegalFedLLM-data\` directory. This supersedes the earlier self-contained EXE
design. WSL, Docker Desktop, Docker Compose and the Linux NVIDIA Container
Toolkit are not part of the Windows Client path.

The physical experiments use **Qwen3 1.7B on the Windows Client** and
**Granite 3.3 2B on the Linux Client**, with a Mistral Nemo Host and normal
trusted quorum 2. Earlier Windows Granite stress runs exposed hard platform
resets under reverse-training load; the current Client memory policy and
experimental model placement were chosen conservatively around the available
VRAM rather than reusing incompatible model/adaptor state.

### Requirements

| Requirement | Why LegalFedLLM needs it | Quick check | Official installation/help |
| --- | --- | --- | --- |
| NVIDIA GPU + working Windows driver | Required by the current real Qwen/Granite Client training path | `nvidia-smi` | [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/) |
| Python 3.14 x64 | Required by the thin Windows launcher's local environment bootstrap | `py -3.14 --version` | [Python for Windows](https://www.python.org/downloads/windows/) |
| OpenSSH Client | Required only when a profile joins the federation through the Client-to-Host SSH tunnel | `ssh -V` | [Microsoft OpenSSH for Windows](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse) |
| Ollama for Windows | Native local-AI/compatibility serving component | `ollama --version` | [Ollama for Windows](https://ollama.com/download/windows) |
| AnythingLLM Desktop for Windows | Native user-facing local RAG/application layer | Check **Settings → Apps → Installed apps**, or launch AnythingLLM | [AnythingLLM Download](https://anythingllm.com/download) |

You will also need enough free disk space for the LegalFedLLM portable data
directory, downloaded model/tokenizer files, Ollama models, and AnythingLLM
application data.

If a quick check already succeeds, do not reinstall that component.

#### NVIDIA driver

A fresh Windows installation may already have a working NVIDIA driver. Check
first:

```powershell
nvidia-smi
```

If the command succeeds and reports the expected NVIDIA GPU, no additional
driver installation is required for this prerequisite.

If the driver is missing or needs to be updated, use NVIDIA's official driver
page:

<https://www.nvidia.com/en-us/drivers/>

The accepted Windows runtime uses the CUDA-capable PyTorch environment verified
by the release bootstrap. Do not install the full CUDA Toolkit merely because
LegalFedLLM uses CUDA-capable PyTorch unless a later runtime explicitly requires
it.

#### OpenSSH Client

Windows provides OpenSSH Client as an optional Windows capability.

Check whether it is already available:

```powershell
ssh -V
```

You can also inspect the Windows capability from PowerShell:

```powershell
Get-WindowsCapability -Online -Name OpenSSH.Client*
```

If it is missing, follow Microsoft's official OpenSSH installation instructions:

<https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse>

Microsoft documents both the **Optional Features** interface and the elevated
PowerShell installation method. Federation connectivity needs the OpenSSH
**Client**, not the OpenSSH Server. A local-only profile can start without
OpenSSH connectivity.

#### Ollama

Check whether Ollama is already installed:

```powershell
ollama --version
```

If it is missing, install the native Windows release from Ollama's official
download page:

<https://ollama.com/download/windows>

Ollama is a serving/compatibility component. It is not the runtime that performs
LegalFedLLM federated PEFT training.

#### AnythingLLM

Check **Settings → Apps → Installed apps** for AnythingLLM, or launch the
AnythingLLM Desktop application if it is already installed.

If it is missing, obtain the native Windows Desktop release from the official
AnythingLLM download page:

<https://anythingllm.com/download>

AnythingLLM is the intended user-facing RAG/application layer. LegalFedLLM
remains responsible for federation, model learning, and its own Client/Host
inference boundary.

#### Explicit model preload

The current Windows source/testing path can preload the exact pinned
Transformers model into the portable LegalFedLLM cache instead of waiting for
the first AnythingLLM `LOCAL` request to trigger a multi-gigabyte download.

From the repository root, first select the active Client model:

```powershell
# Granite 3.3 2B Client
$env:LEGALFEDLLM_MODEL_REPO = "ibm-granite/granite-3.3-2b-instruct"
$env:LEGALFEDLLM_MODEL_REVISION = "652c333dc5066f2a1764854a1bcd0ce67163d74f"
```

or:

```powershell
# Qwen3 1.7B Client
$env:LEGALFEDLLM_MODEL_REPO = "Qwen/Qwen3-1.7B"
$env:LEGALFEDLLM_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
```

Then populate `LegalFedLLM-data\models\huggingface` explicitly:

```powershell
cd C:\path\to\LegalFedLLM

New-Item `
  -ItemType Directory `
  -Force `
  ".\LegalFedLLM-data\models\huggingface" |
  Out-Null

$env:HF_HOME = Join-Path `
  (Get-Location) `
  "LegalFedLLM-data\models\huggingface"

@'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["LEGALFEDLLM_MODEL_REPO"],
    revision=os.environ["LEGALFEDLLM_MODEL_REVISION"],
)

print(f"Model cached at: {path}")
'@ | .\.venv\Scripts\python.exe -
```

Hugging Face reuses completed files and resumes compatible partial downloads in
that cache. Warnings about the optional `hf_xet` package or unavailable Windows
symlinks do not by themselves mean the download failed; wait for the command to
print the final `Model cached at:` path.

The Ollama compatibility copy is separate from the Transformers/PEFT cache.
Install only the model required by the active Client profile:

```powershell
# Granite Client profile
ollama pull granite3.3:2b

# Qwen Client profile
ollama pull qwen3:1.7b
```

The Ollama model does not replace the pinned Transformers model. LegalFedLLM
needs the Transformers/PEFT copy for model-native LoRA training, adapter loading,
validation, and `LOCAL` inference.

### Installation

The Windows release is distributed as a **ZIP archive**, not as a standalone
`LegalFedLLM.exe`. The executable is a thin launcher and must remain beside the
release source/runtime files that it starts.

A normal extracted release looks approximately like this:

```text
LegalFedLLM-Windows-x64\
├── LegalFedLLM.exe
├── client\
├── desktop\
├── legalfed-ai\
├── scripts\
├── shared\
├── requirements.txt
└── requirements-desktop.txt
```

The release archive does **not** include the machine-specific runtime
environment or persistent Client state. They are created beside the release
payload on first use:

```text
LegalFedLLM-Windows-x64\
├── LegalFedLLM.exe
├── .venv\                 ← release-local Python environment
├── LegalFedLLM-data\      ← persistent profiles, adapters, caches and logs
└── ...
```

#### Installing the portable release

1. Install and verify the requirements listed above, especially Python 3.14 x64,
   the NVIDIA driver, Ollama for Windows, and AnythingLLM
   Desktop.
2. Download `LegalFedLLM-Windows-x64.zip` from the project's
   [GitHub Releases](https://github.com/JohnnySusp/LegalFedLLM/releases) page.
3. If the release publishes a SHA-256 checksum, verify the downloaded archive
   against that checksum before extracting it.
4. Extract the **entire archive** into a stable writable directory, for example:

```text
C:\Users\<you>\Applications\LegalFedLLM\
```

Do not run `LegalFedLLM.exe` from inside the ZIP, and do not copy the EXE by
itself to another directory. The launcher expects the bundled LegalFedLLM source
and setup files beside it.

Start LegalFedLLM by double-clicking:

```text
LegalFedLLM.exe
```

On first start the launcher:

```text
checks for Python 3.14 x64
        ↓
creates or reuses .venv\
        ↓
installs/verifies the required CUDA-capable PyTorch and pinned dependencies
        ↓
verifies the Windows runtime
        ↓
starts the PySide6 desktop Client
```

The first start therefore needs Internet access for Python packages unless the
required packages are already available in the local package cache. It can take
substantially longer than later launches because the release-local environment
contains the real Transformers/PEFT/PyTorch Client runtime.

`LegalFedLLM-data\` is created beside the executable and is the persistent
portable state directory. It contains saved profiles, Client identities,
enrollment state, adapters/checkpoints, local-learning state, logs, reference
data, and the release-local Hugging Face model/tokenizer cache.

The native Ollama and AnythingLLM installations remain separate Windows
applications. Their own application/model data is not stored inside
`LegalFedLLM-data\`.

#### Updating an existing Windows release

Close LegalFedLLM before replacing release files.

A release update may replace the launcher and bundled application payload, but
normally preserve:

```text
.venv\
LegalFedLLM-data\
```

`LegalFedLLM-data\` is the important durable state boundary. Back it up before a
manual refresh if the saved Client identities, adapters, or downloaded models
matter.

Because the launcher verifies its release-local environment on startup, a newer
release can update the dependency set in `.venv\` when required. Do not copy an
old `.venv\` into an unrelated clean installation.

### Using LegalFedLLM

#### Starting the Windows Client

Start the application by double-clicking `LegalFedLLM.exe` in the extracted
release directory.

LegalFedLLM opens a launch terminal as part of the Windows workflow. Keep that
terminal open while the Client is running. A fresh profile does **not** open an
SSH connection unless federation enrollment has been requested. When an
enrolled profile connects to the Host, OpenSSH asks for the Host SSH password in
that terminal; LegalFedLLM does not collect or store the SSH password.

A saved enrolled profile still needs the SSH password whenever a new tunnel is
opened, but it does not need another enrollment token after successful
registration.

#### First profile: local use first, optional federation enrollment

Create a Client profile from the GUI and provide:

- a profile name;
- the Qwen or Granite Client model profile;
- the local Coordinator-forward port; and
- the local Client Agent port.

The Host SSH target, SSH port and one-time enrollment token are federation
settings. They may be left blank/unused when the profile is created. In that
state LegalFedLLM starts the Client Agent, local model path and AnythingLLM
integration without opening an SSH tunnel or registering with a Coordinator.

When the user later wants to join a federation, open **Edit current profile…**,
enter the Host SSH target and a fresh one-time enrollment token issued by the
Host/Coordinator, and save the profile. LegalFedLLM then restarts the Client
Agent with the SSH tunnel enabled and performs the existing one-time enrollment
flow.

Each profile is an independent Client identity. It owns its own Client ID,
Ed25519 identity, enrollment state, adapter/checkpoint state, local-learning
queue, and logs.

A successful registration consumes the enrollment token. Do not reuse an old
profile's adapter state for a different model, and do not reuse one enrollment
token for multiple profiles.

Downloaded Transformers model/tokenizer files are shared between profiles
through:

```text
LegalFedLLM-data\models\huggingface\
```

#### Ollama and AnythingLLM

Windows uses the **native** Ollama and AnythingLLM Desktop applications; Docker
Desktop and WSL are not part of the Windows Client path.

When a profile becomes active, LegalFedLLM checks that its expected Ollama
compatibility model is installed. LegalFedLLM does not silently download a
missing Ollama model. Install the model explicitly if required:

```powershell
# Qwen Client profile
ollama pull qwen3:1.7b

# Granite Client profile
ollama pull granite3.3:2b
```

The Ollama copy is separate from the pinned Transformers/PEFT model used for
federated training and `LOCAL` inference.

After the Client Agent is healthy, LegalFedLLM detects or launches AnythingLLM
Desktop and configures its reserved Generic OpenAI connection for the active
LegalFedLLM profile. On a fresh AnythingLLM Desktop installation whose onboarding
has not yet been completed, LegalFedLLM selects **Generic OpenAI**, configures the
active profile's loopback base URL and API key, selects `legalfedllm-local`, and
marks AnythingLLM's one-time onboarding complete before bringing the AnythingLLM
Desktop UI forward. The user therefore does not need to open AnythingLLM first or
complete its provider setup manually.

For an already-onboarded AnythingLLM installation, LegalFedLLM refreshes the
reserved Generic OpenAI connection but preserves the existing default provider.
The Windows LegalFedLLM GUI exposes an **AnythingLLM model** control with
**Local** and **Host** choices. Selecting either choice is an explicit user action
that switches AnythingLLM to LegalFedLLM's Generic OpenAI provider and the
corresponding model. **Host** is accepted only while the profile is enrolled and
the Coordinator connection is active. AnythingLLM owns the user-facing chat/RAG
experience; LegalFedLLM owns federation, model learning, Client identity, and the
`legalfedllm-local` / `legalfedllm-host` model boundary.

The desktop's **AnythingLLM integration details** dialog remains a diagnostic and
manual-fallback view. It shows the provider, base URL, API key, one `Model` value
(defaulting to `legalfedllm-local`), context/max-token settings, and required
Ollama compatibility model as individually copyable read-only fields. Internal
runtime-directory paths are not shown in this user-facing dialog.

AnythingLLM can take additional time to initialize on first launch. Its local
backend normally listens on:

```text
http://127.0.0.1:3001/
```

#### Normal federation use

The main GUI action is **Participate in the current federated round**. It becomes
available only when the active Client is connected, compatible, selected for a
collecting round, and has not already participated.

Before participating, verify that:

- the GUI shows the intended saved profile and model;
- the Coordinator connection is healthy;
- the displayed round is the round you intend to join; and
- the required Ollama compatibility model is installed.

If queued local learning exists, LegalFedLLM applies it according to the active
desktop settings before creating the Client Knowledge Package. When a completed
round contains usable Host-to-Client teaching samples, reverse learning requires
the Client-side consent/validation path before any candidate is adopted.

The global Options menu includes:

- **Constant Learning** — enabled by default; LOCAL interactions enter the
  one-use local-learning queue automatically;
- **Low VRAM Mode** — enabled by default. CUDA Clients always retain the baseline
  memory protections; Low VRAM Mode further reduces reference/validation chunks
  from 64 to 32 tokens and requests expandable CUDA allocator segments on Linux.
  Changing the setting saves the next-launch value but keeps the current Client
  running with its existing effective memory settings. Restart LegalFedLLM
  manually when convenient to apply the change;
- **Debug Mode** — opens additional Client/GPU diagnostic terminals; and
- **Reset Defaults** — restores the default desktop settings.

For constrained GPUs, Low VRAM Mode can reduce reference/validation memory
pressure, but it does not turn unsupported hardware into a guaranteed-safe
training target.

#### Closing LegalFedLLM

Close the GUI normally. LegalFedLLM stops its Client Agent and SSH tunnel while
preserving the saved profile, adapter/model state, queues, logs, and caches
under `LegalFedLLM-data\`.

AnythingLLM Desktop and Ollama are native third-party applications. Keep the
LegalFedLLM launch terminal open while the Windows Client is running. In the
current Windows release, terminating that terminal can also terminate native
AnythingLLM/Ollama processes launched from the same console session. Their
persistent application/model data remains owned by those applications rather
than by `LegalFedLLM-data\`.

### Uninstallation

The Windows release is portable: there is no separate LegalFedLLM installer to
remove.

First close LegalFedLLM and make sure its launch terminal and Client Agent have
exited. If you may want to restore the same Client identities, adapters, or model
cache later, back up:

```text
LegalFedLLM-data\
```

For a complete LegalFedLLM removal, delete the entire extracted release
directory, including:

```text
LegalFedLLM.exe
client\
desktop\
legalfed-ai\
scripts\
shared\
requirements.txt
requirements-desktop.txt
.venv\
LegalFedLLM-data\
```

Deleting `LegalFedLLM-data\` is destructive: it removes the saved Client
profiles, private Client identities/keys, enrollment state, adapters/checkpoints,
local-learning state, logs, and the LegalFedLLM Hugging Face cache.

The following prerequisites are installed independently of LegalFedLLM and are
**not** removed when the portable release directory is deleted:

- the NVIDIA driver;
- Python 3.14 x64;
- Windows OpenSSH Client;
- Ollama for Windows; and
- AnythingLLM Desktop.

Remove those separately through their normal Windows/official uninstall methods
only if they are no longer needed by other applications.

Ollama's model store is also outside `LegalFedLLM-data\`. If you want to remove
only the compatibility models that were installed for LegalFedLLM, inspect the
installed models first:

```powershell
ollama list
```

Then remove only the models you no longer want, for example:

```powershell
ollama rm qwen3:1.7b
ollama rm granite3.3:2b
```

AnythingLLM owns its own application data and should be cleaned up through
AnythingLLM's normal uninstall/data-management path rather than by deleting
LegalFedLLM files.

LegalFedLLM does not store the SSH password. Windows OpenSSH may retain the
Host's public key in the user's normal `known_hosts` file. If you deliberately
want to remove that record as well, use `ssh-keygen -R` with the Host name/IP
and the appropriate `[host]:port` form for a non-default port.

After the extracted release directory is deleted, and after any optional
third-party model/application cleanup above, the current portable Windows path
does not require another LegalFedLLM-specific OS-global application-data
directory.


## Linux

### Requirements

LegalFedLLM v1.0.0 can be run on Linux either from the published **x86_64
AppImage** or directly from source. The AppImage packages the desktop
GUI/controller, but it deliberately does not install host-level prerequisites
such as Docker, OpenSSH, or the NVIDIA runtime.

#### Runtime requirements

| Requirement | AppImage | Source | Quick check |
| --- | --- | --- | --- |
| Linux x86_64 | Required by the current published AppImage | Current tested desktop platform | `uname -m` |
| Docker Engine | Required | Required for the managed local-AI stack | `docker --version` |
| Docker Compose | Required | Required for the managed local-AI stack | `docker compose version` |
| OpenSSH client | Required only for federation | Required only for federation | `ssh -V` |
| NVIDIA GPU + working Linux driver | Required by the current real Qwen/Granite Client path | Required by the current real Qwen/Granite Client path | `nvidia-smi` |
| NVIDIA Container Toolkit / Docker GPU runtime | Required by the Dockerized real-model path | Required when the managed Docker services use the GPU | `docker run --rm --gpus all ubuntu nvidia-smi` |
| Git | Not required | Required to clone/update the source checkout | `git --version` |
| Python 3 with `venv` and `pip` | Not required | Required | `python3 --version` |

You also need enough free disk space for Docker images and downloaded
model/tokenizer data. The first AppImage launch builds a versioned
`legalfedllm-client:<runtime-hash>` image locally, so Docker storage usage is
larger than the AppImage file itself.

If a command in the table already works, do not reinstall that component.

#### Installing the required host tools

##### Docker Engine and Docker Compose

On a normal mutable Linux distribution, use Docker Engine rather than relying on
the AppImage to provide Docker. Docker publishes distribution-specific
instructions for [Ubuntu](https://docs.docker.com/engine/install/ubuntu/),
[Debian](https://docs.docker.com/engine/install/debian/), and
[Fedora](https://docs.docker.com/engine/install/fedora/).

For Fedora, after adding Docker's official repository as documented above:

```bash
sudo dnf config-manager addrepo --from-repofile \
  https://download.docker.com/linux/fedora/docker-ce.repo
sudo dnf install docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```

For Ubuntu or Debian, after adding Docker's official `apt` repository as
documented for the distribution:

```bash
sudo apt install docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
```

On Fedora Silverblue and Silverblue-derived immutable systems, a Fedora-native
host installation can instead be layered with `rpm-ostree`:

```bash
sudo rpm-ostree install moby-engine docker-cli docker-compose
systemctl reboot
sudo systemctl enable --now docker
```

Bazzite and other image-based systems may already include some of these
components. Check first and layer only what is missing.

LegalFedLLM launches Docker as the current desktop user, so Docker must work
without prefixing every command with `sudo`. If your Docker installation uses
the normal `docker` group:

```bash
sudo usermod -aG docker "$USER"
```

Sign out and back in after changing group membership, then verify:

```bash
docker --version
docker compose version
docker run --rm hello-world
```

> **Security note:** membership in the `docker` group grants root-level
> privileges through the Docker daemon. See Docker's
> [Linux post-installation guidance](https://docs.docker.com/engine/install/linux-postinstall/)
> before enabling it on a shared machine.

##### OpenSSH client

This section is only needed when the Client will join a federation. Local-only
profiles do not require an SSH tunnel.

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install openssh-client
```

On mutable Fedora:

```bash
sudo dnf install openssh-clients
```

On Fedora Silverblue/Silverblue-derived systems, if `ssh` is not already
present:

```bash
sudo rpm-ostree install openssh-clients
systemctl reboot
```

Verify with:

```bash
ssh -V
```

LegalFedLLM deliberately leaves SSH password entry to OpenSSH. It does not
handle or store the Host SSH password.

##### NVIDIA driver and NVIDIA Container Toolkit

The current real Qwen and Granite Client paths assume an NVIDIA/CUDA-capable
Linux environment. First install a working NVIDIA driver using the supported
package/image method for your distribution, then verify:

```bash
nvidia-smi
```

For normal mutable distributions, install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
using NVIDIA's repository for your distribution. On Fedora/RPM-based systems,
the short form after adding NVIDIA's repository is:

```bash
curl -s -L \
  https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

sudo dnf install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

For Ubuntu/Debian, use the `apt` repository setup in NVIDIA's installation
guide, install `nvidia-container-toolkit`, then run the same
`nvidia-ctk runtime configure --runtime=docker` command and restart Docker.

On **Bazzite**, use the appropriate NVIDIA Bazzite image for the GPU; Bazzite's
NVIDIA images already carry and update the NVIDIA driver. If the NVIDIA
Container Toolkit itself is missing, add NVIDIA's RPM repository and layer the
toolkit:

```bash
curl -s -L \
  https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

sudo rpm-ostree install nvidia-container-toolkit
systemctl reboot
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

For other Silverblue-derived systems, make sure the host NVIDIA driver is
working before layering the Container Toolkit.

Finally verify Docker GPU access:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

##### Git

Git is needed only for the source installation/development path.

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install git
```

On mutable Fedora:

```bash
sudo dnf install git
```

On Fedora Silverblue/Silverblue-derived systems, if Git is not already present:

```bash
sudo rpm-ostree install git
systemctl reboot
```

Verify with:

```bash
git --version
```

##### Python 3, `venv`, and `pip`

Python is needed only for the source installation/development path. The
published AppImage does not require a separate LegalFedLLM Python environment.

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

On mutable Fedora:

```bash
sudo dnf install python3 python3-pip
```

On Fedora Silverblue/Silverblue-derived systems, if the required Python tools
are not already present:

```bash
sudo rpm-ostree install python3 python3-pip
systemctl reboot
```

Verify that virtual environments work:

```bash
python3 --version
python3 -m venv --help >/dev/null
```

The LegalFedLLM Python libraries do **not** need to be installed one by one.
`requirements-desktop.txt` includes the normal runtime requirements plus
PySide6 and PyInstaller. The Installation section below installs the pinned set
into a project-local virtual environment.

##### `appimagetool` — build-only

`appimagetool` is **not** required to run the published AppImage or to run
LegalFedLLM from source. It is required only when building the Linux AppImage
with:

```bash
python scripts/build_desktop.py --appimage
```

Download the current x86_64 binary from the
[`AppImage/appimagetool` releases](https://github.com/AppImage/appimagetool/releases),
make it executable, and place it somewhere on `PATH`. A user-local installation
works on both mutable and immutable Linux:

```bash
mkdir -p ~/.local/bin
install -m 0755 appimagetool-x86_64.AppImage ~/.local/bin/appimagetool
```

If `~/.local/bin` is not already on `PATH`, add it in your shell configuration.
No host package layering is required.

### Installation

The **Linux x86_64 AppImage is the recommended end-user installation path** for
the v1.0.0 release. The source path remains available for development,
inspection, and direct source execution.

#### Method 1 — Linux x86_64 AppImage

Download `LegalFedLLM-x86_64.AppImage` from the project's
[GitHub Releases](https://github.com/JohnnySusp/LegalFedLLM/releases) page and
place it in a stable directory before first use. For example:

```bash
mkdir -p ~/Applications/LegalFedLLM
mv ~/Downloads/LegalFedLLM-x86_64.AppImage ~/Applications/LegalFedLLM/
chmod +x ~/Applications/LegalFedLLM/LegalFedLLM-x86_64.AppImage
```

The release page publishes the SHA-256 checksum for the AppImage. Before first
use, compare the downloaded file against the checksum shown for that release.

The AppImage is intentionally lightweight: it contains the PySide6
GUI/controller, SSH-tunnel control, profile management, local-AI orchestration,
and the release-specific Client runtime definition. On first use it materializes
the Client runtime below the portable data directory and builds a versioned
`legalfedllm-client:<runtime-hash>` Docker image locally. Later launches reuse
the matching image.

##### AppImage data location

By default, persistent LegalFedLLM desktop state is stored next to the AppImage:

```text
~/Applications/LegalFedLLM/
├── LegalFedLLM-x86_64.AppImage
└── LegalFedLLM-data/
    ├── desktop-state.json
    ├── profiles/
    │   └── profile-.../
    │       ├── profile.json
    │       ├── profile.env
    │       ├── client-data/
    │       ├── private/
    │       └── logs/
    ├── models/
    │   └── huggingface/
    ├── client-runtime/
    │   └── <runtime-hash>/
    └── legalfed-ai/
```

The important rule is:

```text
AppImage directory
├── LegalFedLLM-x86_64.AppImage
└── LegalFedLLM-data/   ← persistent LegalFedLLM desktop state
```

The profile directories contain Client identity/enrollment state, adapters and
checkpoints, local-learning queue, and logs. `models/huggingface/` is the shared
Hugging Face cache. `client-runtime/` contains the materialized Docker Client
runtime. `legalfed-ai/` contains the writable Ollama/AnythingLLM Compose
configuration used by the desktop.

If you move the AppImage and want to preserve the same profiles, move its
sibling `LegalFedLLM-data/` directory with it. Moving only the AppImage makes the
new directory look like a fresh installation.

#### Method 2 — source checkout

Clone the repository, create a project-local virtual environment, and install
the desktop requirements:

```bash
git clone https://github.com/JohnnySusp/LegalFedLLM.git
cd LegalFedLLM

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-desktop.txt
```

`requirements-desktop.txt` includes the normal LegalFedLLM runtime requirements
plus PySide6 and the desktop build dependencies. The first source installation
can therefore be large because the local environment contains the
Transformers/PEFT/PyTorch Client stack.

Before first launch, verify the host-level prerequisites:

```bash
docker --version
docker compose version
ssh -V
nvidia-smi
```

##### Source-mode data location

When run from source, persistent desktop state is created inside the checkout:

```text
LegalFedLLM/
├── ...
└── LegalFedLLM-data/
```

Do not delete `LegalFedLLM-data/` if you want to preserve Client profiles,
identities, adapters/checkpoints, local-learning state, logs, model/tokenizer
downloads, and the managed local-AI runtime copy.

### Using LegalFedLLM

Both installation methods use the same saved-profile, local-AI, and federation
workflow.

#### Starting LegalFedLLM

For a source installation, start the desktop Client from the repository root:

```bash
cd LegalFedLLM
source .venv/bin/activate
python -m desktop.app
```

Keep that terminal open. Local-only profiles do not open an SSH connection.
When federation connectivity is enabled for a profile, LegalFedLLM leaves SSH
password entry to OpenSSH, so the Host SSH password is entered in the launch
terminal and is not handled or stored by LegalFedLLM.

For an AppImage installation, double-click the AppImage in a file manager or
launch it from a terminal:

```bash
~/Applications/LegalFedLLM/LegalFedLLM-x86_64.AppImage
```

When launched graphically on Linux, the AppImage opens a terminal and starts the
GUI from that terminal. A local-only profile does not invoke OpenSSH; the same
terminal is used for Host password entry only after federation connectivity is
enabled. If the desktop environment cannot provide one of the supported
terminal launchers, start the AppImage from an existing terminal.

#### First profile: local use first, optional federation enrollment

On first launch, create a Client profile in the GUI and provide a profile name,
the Qwen or Granite Client model profile, and the local Coordinator-forward and
Client Agent ports. The Host SSH target and one-time enrollment token are
optional at creation time.

With no enrollment token, the profile starts in **local-only mode**: the Client
Agent and local AI stack start without an SSH tunnel, Coordinator registration,
or federated-round participation. To join a federation later, use **Edit current
profile…** to add the Host SSH target and a fresh one-time enrollment token.
Successful registration consumes that token. Later launches of the same
enrolled profile use its persisted Client identity and do not require a new
enrollment token.

Each profile owns an independent Client ID, Ed25519 identity, adapter/checkpoint
state, private-learning queue, and logs. Downloaded Hugging Face model/tokenizer
data is shared through `LegalFedLLM-data/models/huggingface/`.

#### Ollama and AnythingLLM on first use

After the Client Agent becomes healthy, LegalFedLLM prepares the managed Docker
Ollama and AnythingLLM stack. In local-only mode this happens without an SSH
tunnel; enrolled profiles may establish their Host tunnel in parallel with the
federation path. Docker may pull those service images if they are not already
available. When the local AI
stack is ready, the desktop opens AnythingLLM in the default browser. The
**AnythingLLM integration details** dialog exposes the provider connection as
individual read-only fields with **Copy** buttons for manual setup or inspection;
it does not expose the internal runtime-files directory.

LegalFedLLM does **not** silently pull the selected Ollama compatibility model.
If the GUI reports that the required model is missing after Ollama starts,
install it explicitly:

```bash
# Qwen Client profile
docker exec legalfed-ai-ollama ollama pull qwen3:1.7b

# Granite Client profile
docker exec legalfed-ai-ollama ollama pull granite3.3:2b
```

Only the model required by the active Client profile needs to be installed.
Federated training and LOCAL LegalFedLLM inference use the exact pinned
Transformers + PEFT Client state; Ollama remains a compatibility/local-AI
serving component rather than a substitute for the active LegalFedLLM adapter.

For a **source checkout**, the same pinned Transformers model can also be
preloaded explicitly into the shared LegalFedLLM cache before the first
model-backed request. Select one active Client profile:

```bash
# Granite 3.3 2B Client
export LEGALFEDLLM_MODEL_REPO='ibm-granite/granite-3.3-2b-instruct'
export LEGALFEDLLM_MODEL_REVISION='652c333dc5066f2a1764854a1bcd0ce67163d74f'

# Or, for Qwen3 1.7B:
# export LEGALFEDLLM_MODEL_REPO='Qwen/Qwen3-1.7B'
# export LEGALFEDLLM_MODEL_REVISION='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
```

Then run:

```bash
cd /path/to/LegalFedLLM
source .venv/bin/activate
mkdir -p LegalFedLLM-data/models/huggingface
export HF_HOME="$PWD/LegalFedLLM-data/models/huggingface"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["LEGALFEDLLM_MODEL_REPO"],
    revision=os.environ["LEGALFEDLLM_MODEL_REVISION"],
)

print(f"Model cached at: {path}")
PY
```

The source-mode command above is optional; normal LegalFedLLM model loading will
also populate the same cache as needed.

#### Normal use

On later launches, start LegalFedLLM using the same source or AppImage command
from **Starting LegalFedLLM**, then select the saved profile.

A local-only profile starts directly without SSH. A previously enrolled profile
still needs the SSH password for the new tunnel connection, but it does not need
another enrollment token. After the Client Agent and local AI services are
ready, AnythingLLM opens automatically.

The OpenAI-compatible Client endpoint exposes the `legalfedllm-local` and
`legalfedllm-host` routes used by the desktop integration. `LOCAL` stays on the
Client and uses the active Client PEFT state. `HOST` explicitly forwards through
the LegalFedLLM Host/Coordinator path.

The main federation action is **Participate in the current federated round**. It
is available only when the active Client is connected, compatible, selected for
a collecting round, and has not already participated. Queued LOCAL learning is
applied before participation when present.

The global desktop settings are available from the Options menu:

- **Constant Learning** is enabled by default. LOCAL interactions are added to
  the one-use local-learning queue automatically. Disable it to restore the
  explicit **Learn from this** / **Dismiss** decision.
- **Low VRAM Mode** is enabled by default. CUDA Clients always use checkpointed
  training and bounded reference/validation inference. Enabling this option
  reduces reference/validation chunks from 64 to 32 tokens and requests expandable
  allocator segments on Linux. Changing it saves the next-launch value while the
  current Client stays open with its existing effective settings; restart
  LegalFedLLM manually when convenient to apply the change.
- **Debug Mode** is disabled by default. Enable it to open the additional Client
  state and NVIDIA/GPU diagnostic terminals.
- **Reset Defaults** restores Constant Learning to on, Debug Mode to off, and
  Low VRAM Mode to on. If that changes the effective Low VRAM setting, the GUI
  confirms that a manual restart is required but does not close the running
  Client automatically.

When a completed round contains useful Host-to-Client teaching samples,
LegalFedLLM asks before applying reverse learning.

#### Closing LegalFedLLM

Closing the GUI stops the Client Agent and its SSH tunnel. In the AppImage path,
the profile-specific LegalFedLLM Client Docker container is also brought down;
the Client's persistent profile state, model cache, and locally built Docker
image are retained for later launches.

If managed Ollama or AnythingLLM services are actually running, the GUI asks
whether to stop them or leave them running. Choosing **Stop Docker and Close**
stops those services while preserving their Docker volumes. If both services
are already stopped, the desktop closes without that prompt.

### Uninstallation

#### AppImage and desktop Docker resources

A normal AppImage installation has LegalFedLLM-specific persistent state in two
places:

1. the directory containing the AppImage and its sibling `LegalFedLLM-data/`;
2. Docker resources created or used by the desktop.

First close LegalFedLLM. If the Ollama/AnythingLLM shutdown prompt appears,
choose **Stop Docker and Close**.

Then delete the AppImage and its sibling portable data directory. For the
example layout above:

```bash
rm -f ~/Applications/LegalFedLLM/LegalFedLLM-x86_64.AppImage
rm -rf ~/Applications/LegalFedLLM/LegalFedLLM-data
```

Those commands remove the AppImage and filesystem state owned by that portable
installation. Adjust the path if you stored the AppImage elsewhere.

If you also want to remove the Docker resources associated with LegalFedLLM,
inspect them first:

```bash
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' \
  | grep -E '(^NAMES|legalfedllm-|legalfed-ai-)'
docker image ls legalfedllm-client
docker volume ls | grep 'legalfed-ai-'
docker network ls | grep 'legalfed-ai-net'
```

Then, for a complete LegalFedLLM-specific Docker cleanup:

```bash
# Remove any remaining profile Client containers.
docker ps -a --format '{{.ID}} {{.Names}}' \
  | awk '$2 ~ /^legalfedllm-profile-.*-client-1$/ {print $1}' \
  | xargs -r docker rm -f

# Remove the managed local-AI containers.
docker rm -f \
  legalfed-ai-anythingllm \
  legalfed-ai-ollama \
  2>/dev/null || true

# Remove persistent AnythingLLM and Ollama data.
docker volume rm \
  legalfed-ai-anythingllm-data \
  legalfed-ai-ollama-data \
  2>/dev/null || true

# Remove the LegalFedLLM local-AI network.
docker network rm legalfed-ai-net 2>/dev/null || true

# Remove locally built LegalFedLLM Client runtime images.
docker image ls legalfedllm-client -q \
  | sort -u \
  | xargs -r docker image rm
```

The two `legalfed-ai-*` volumes contain persistent AnythingLLM state and Ollama
model data. Removing them is destructive and should only be done for a complete
uninstall.

The generic third-party Ollama and AnythingLLM image layers may remain in
Docker's shared image cache, and Docker can also retain shared build cache.
LegalFedLLM intentionally does not recommend broad commands such as
`docker system prune` or `docker builder prune` as part of its uninstall path,
because those can delete resources belonging to unrelated projects.

LegalFedLLM does not store the SSH password. OpenSSH may add the Host key to the
user's normal `~/.ssh/known_hosts`; that file belongs to OpenSSH rather than
LegalFedLLM. A user who also wants to remove that Host-key record can use
`ssh-keygen -R <host>` and, for a non-default SSH port, the corresponding
`[host]:port` form.

With the AppImage file, its sibling `LegalFedLLM-data/`, the LegalFedLLM-specific
Docker containers/images/volumes/network, and any deliberately removed OpenSSH
host-key record gone, the current portable AppImage path does not require any
other OS-global LegalFedLLM application-data directory.

#### Source checkout

To completely remove a source installation, remove the repository checkout
including its `LegalFedLLM-data/` directory. If you also want to remove the
managed Docker state created by the desktop, use the LegalFedLLM-specific Docker
cleanup commands above. Do not use broad Docker prune commands unless you
intend to remove resources belonging to other projects too.

## Host

The Host and Coordinator run separately from desktop Clients. The commands in
this section are written for the established A40 container deployment and are
intended to be copied directly into the Host shell. They intentionally omit the
external SSH address, username and provider-specific connection details.

All operational blocks below are Bash functions that are defined and then called in
the same copy-paste block. Errors use `return 1` rather than `exit 1`, so a failed
check returns control to the current SSH prompt instead of terminating the login
shell. The functions deliberately do not enable `set -e`.

> **Host deployment boundary:** `/scratch/legalfedllm-test/` and everything
> beneath it are deployment-specific, private Host state. The public source
> archive does **not** contain or recreate the deployed checkout, shared virtual
> environment, `.env.host`, private datasets, downloaded models/caches,
> artifacts, logs or prior round state stored there. Do not copy, publish,
> delete, regenerate or replace the contents of `/scratch/legalfedllm-test/`
> merely because this README documents the paths. These commands apply to the
> established Host installation only; a different Host must first be provisioned
> with an equivalent deployment layout.

The documented Host layout is:

```text
/scratch/legalfedllm-test/
├── .venv/                       shared Host Python environment
├── work/
│   └── LegalFedLLM/             deployed source checkout
│       ├── .env.host            private Host/Coordinator environment
│       ├── host/
│       ├── coordinator/
│       ├── shared/
│       └── scripts/
│           ├── run_host_stack.py
│           ├── issue_enrollment_token.py
│           └── create_remote_round.py
├── datasets/
│   └── gld2012/
│       ├── reference.jsonl      private D^P reference data
│       └── validation.jsonl     private Host-only D^V validation data
├── cache/                       Hugging Face, pip, Torch, Triton, XDG, CUDA
├── tmp/
├── logs/
├── artifacts/                   Host and Coordinator runtime state
└── incoming/                    deployment staging
```

### Enter the existing Host installation

Run this at the beginning of a fresh Host shell. It fails immediately if the
expected deployed repository, virtual environment or private Host environment is
missing, preventing later commands from accidentally running in `/home/iosider`
or against the system Python installation.

```bash
legalfed_host_enter() {
  local root=/scratch/legalfedllm-test
  local repo="$root/work/LegalFedLLM"
  local activate="$root/.venv/bin/activate"

  test -d "$repo" || {
    echo "ERROR: $repo is missing"
    return 1
  }
  test -f "$activate" || {
    echo "ERROR: $root/.venv is missing"
    return 1
  }

  cd "$repo" || {
    echo "ERROR: could not enter $repo"
    return 1
  }
  source "$activate" || {
    echo "ERROR: could not activate $root/.venv"
    return 1
  }

  export LEGALFEDLLM_TEST_ROOT="$root"
  export HF_HOME="$root/cache/huggingface"
  export PIP_CACHE_DIR="$root/cache/pip"
  export TORCH_HOME="$root/cache/torch"
  export TORCH_EXTENSIONS_DIR="$root/cache/torch-extensions"
  export TRITON_HOME="$root/cache/triton"
  export XDG_CACHE_HOME="$root/cache/xdg"
  export CUDA_CACHE_PATH="$root/cache/cuda"
  export TMPDIR="$root/tmp"
  export TOKENIZERS_PARALLELISM=false

  mkdir -p \
    "$HF_HOME" \
    "$PIP_CACHE_DIR" \
    "$TORCH_HOME" \
    "$TORCH_EXTENSIONS_DIR" \
    "$TRITON_HOME" \
    "$XDG_CACHE_HOME" \
    "$CUDA_CACHE_PATH" \
    "$TMPDIR" \
    "$root/logs" || {
      echo 'ERROR: could not create one or more Host cache/runtime directories'
      return 1
    }

  local required
  for required in \
    requirements.txt \
    scripts/run_host_stack.py \
    scripts/issue_enrollment_token.py \
    scripts/create_remote_round.py \
    .env.host
  do
    test -f "$required" || {
      echo "ERROR: $required is missing"
      return 1
    }
  done

  printf 'repo=%s\npython=%s\n' "$PWD" "$(command -v python)"
  python --version || return 1
  nvidia-smi || return 1
}

legalfed_host_enter
```

The normal Host installation uses the existing shared environment at
`/scratch/legalfedllm-test/.venv`. Do not create another `.venv` inside the
repository as part of routine Host startup.

### Fresh Host bootstrap only

Use this only when the source tree and shared venv have already been provisioned
but `.env.host` has not yet been created. It is **not** a normal startup command
for an existing Host.

```bash
legalfed_host_bootstrap() {
  local root=/scratch/legalfedllm-test
  local repo="$root/work/LegalFedLLM"
  local activate="$root/.venv/bin/activate"

  test -d "$repo" || { echo "ERROR: $repo is missing"; return 1; }
  test -f "$activate" || { echo "ERROR: $root/.venv is missing"; return 1; }
  cd "$repo" || { echo "ERROR: could not enter $repo"; return 1; }
  source "$activate" || { echo "ERROR: could not activate $root/.venv"; return 1; }

  export PIP_CACHE_DIR="$root/cache/pip"
  export TMPDIR="$root/tmp"
  mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" || return 1

  python -m pip install -r requirements.txt || {
    echo 'ERROR: dependency installation failed'
    return 1
  }

  test ! -e .env.host || {
    echo 'ERROR: .env.host already exists; inspect and reuse it instead of replacing it'
    return 1
  }

  python scripts/bootstrap.py host \
    --output .env.host \
    --runtime-root "$root" || {
      echo 'ERROR: Host bootstrap failed'
      return 1
    }
}

legalfed_host_bootstrap
```

Host bootstrap creates private administrative/internal tokens and Host runtime
paths. It does not download or synthesize the private GLD-derived D^P/D^V files.
Those datasets must already be available at the paths configured in `.env.host`.
D^V remains Host/Coordinator-only.

### Check whether Host and Coordinator are running

The Host API is expected on `127.0.0.1:8002` and the Coordinator on
`127.0.0.1:8000`.

```bash
legalfed_host_status() {
  local repo=/scratch/legalfedllm-test/work/LegalFedLLM
  local activate=/scratch/legalfedllm-test/.venv/bin/activate

  test -d "$repo" || { echo "ERROR: $repo is missing"; return 1; }
  test -f "$activate" || { echo 'ERROR: Host virtual environment is missing'; return 1; }
  cd "$repo" || return 1
  source "$activate" || return 1

  echo '=== LegalFedLLM processes ==='
  ps -ef | grep -E \
    'run_host_stack.py|uvicorn.*host.main|uvicorn.*coordinator.main' \
    | grep -v grep || true

  echo
  echo '=== Host health :8002 ==='
  if curl -fsS --max-time 5 http://127.0.0.1:8002/health; then
    echo
    echo 'Host is healthy'
  else
    echo
    echo 'Host is not reachable'
  fi

  echo
  echo '=== Coordinator health :8000 ==='
  if curl -fsS --max-time 5 http://127.0.0.1:8000/health; then
    echo
    echo 'Coordinator is healthy'
  else
    echo
    echo 'Coordinator is not reachable'
  fi

  echo
  nvidia-smi || return 1
}

legalfed_host_status
```

Do not infer service state from `ss`, `lsof` or `fuser` alone. Process inspection
plus the direct health probes above are the normal checks for this deployment.

### Ensure the normal two-Client Host/Coordinator stack is running

This is the normal federation mode. It requires majority quorum with minimum
trusted quorum `2` and no one-Client override.

The block below is safe to run when the stack is already healthy: it reports the
current Coordinator quorum configuration and does not start a duplicate stack.
If neither service is healthy, it starts the stack detached and waits for both
health endpoints. If only one of the two services is healthy, it stops rather
than guessing about a partial stack.

```bash
legalfed_host_start_normal() {
  local repo=/scratch/legalfedllm-test/work/LegalFedLLM
  local activate=/scratch/legalfedllm-test/.venv/bin/activate
  local log=/scratch/legalfedllm-test/logs/legalfedllm-host-stack.log
  local pidfile=/scratch/legalfedllm-test/logs/legalfedllm-host-stack.pid
  local host_ok=0 coord_ok=0 coord_health i

  test -d "$repo" || { echo "ERROR: $repo is missing"; return 1; }
  test -f "$activate" || { echo 'ERROR: Host virtual environment is missing'; return 1; }
  cd "$repo" || return 1
  source "$activate" || return 1

  curl -fsS --max-time 5 http://127.0.0.1:8002/health >/dev/null 2>&1 && host_ok=1
  curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && coord_ok=1

  if [ "$host_ok" -eq 1 ] && [ "$coord_ok" -eq 1 ]; then
    echo 'Host and Coordinator are already healthy.'
    coord_health="$(curl -fsS http://127.0.0.1:8000/health)" || return 1
    printf '%s\n' "$coord_health" | python -m json.tool || return 1
    printf '%s\n' "$coord_health" | python -c '
import json, sys
h=json.load(sys.stdin)
if h.get("quorum_policy") != "majority":
    raise SystemExit("ERROR: Coordinator is not using majority quorum")
if str(h.get("minimum_trusted_client_quorum")) != "2":
    raise SystemExit("ERROR: Coordinator minimum trusted quorum is not 2")
if str(h.get("trusted_client_quorum_override")) not in {"none", "", "None"}:
    raise SystemExit("ERROR: Coordinator is running with a one-Client override; stop it before normal two-Client use")
print("Normal two-Client quorum configuration is active.")
' || return 1
    return 0
  fi

  if [ "$host_ok" -eq 1 ] || [ "$coord_ok" -eq 1 ]; then
    echo 'ERROR: only one of Host/Coordinator is healthy. Inspect the partial stack before starting anything else.'
    ps -ef | grep -E \
      'run_host_stack.py|uvicorn.*host.main|uvicorn.*coordinator.main' \
      | grep -v grep || true
    return 1
  fi

  export COORDINATOR_MINIMUM_TRUSTED_CLIENT_QUORUM=2
  export COORDINATOR_TRUSTED_CLIENT_QUORUM_OVERRIDE=""
  mkdir -p /scratch/legalfedllm-test/logs || return 1

  nohup python scripts/run_host_stack.py \
    --env-file .env.host \
    >"$log" 2>&1 < /dev/null &

  echo $! > "$pidfile"
  echo "Started Host stack parent PID $(cat "$pidfile")"
  echo "Log: $log"

  for i in $(seq 1 180); do
    host_ok=0
    coord_ok=0
    curl -fsS --max-time 2 http://127.0.0.1:8002/health >/dev/null 2>&1 && host_ok=1
    curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && coord_ok=1
    if [ "$host_ok" -eq 1 ] && [ "$coord_ok" -eq 1 ]; then
      break
    fi
    if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
      echo 'ERROR: Host stack exited during startup'
      tail -n 120 "$log"
      return 1
    fi
    sleep 5
  done

  curl -fsS --max-time 5 http://127.0.0.1:8002/health || {
    echo 'ERROR: Host did not become healthy'
    tail -n 120 "$log"
    return 1
  }
  echo

  coord_health="$(curl -fsS --max-time 5 http://127.0.0.1:8000/health)" || {
    echo 'ERROR: Coordinator did not become healthy'
    tail -n 120 "$log"
    return 1
  }
  printf '%s\n' "$coord_health" | python -m json.tool || return 1
  printf '%s\n' "$coord_health" | python -c '
import json, sys
h=json.load(sys.stdin)
assert h.get("quorum_policy") == "majority", h
assert str(h.get("minimum_trusted_client_quorum")) == "2", h
assert str(h.get("trusted_client_quorum_override")) in {"none", "", "None"}, h
print("Normal two-Client Host/Coordinator stack is ready.")
' || return 1
}

legalfed_host_start_normal
```

### Ensure the one-Client test Host/Coordinator stack is running

A one-Client round is a controlled proof-of-concept/testing mode. The running
Coordinator must itself be started with trusted quorum override `1`; setting the
override only on `create_remote_round.py` is insufficient because the
Coordinator resolves quorum using its own startup configuration.

Do not use this override for normal two-Client federation.

```bash
legalfed_host_start_single_client() {
  local repo=/scratch/legalfedllm-test/work/LegalFedLLM
  local activate=/scratch/legalfedllm-test/.venv/bin/activate
  local log=/scratch/legalfedllm-test/logs/legalfedllm-host-stack.log
  local pidfile=/scratch/legalfedllm-test/logs/legalfedllm-host-stack.pid
  local host_ok=0 coord_ok=0 coord_health i

  test -d "$repo" || { echo "ERROR: $repo is missing"; return 1; }
  test -f "$activate" || { echo 'ERROR: Host virtual environment is missing'; return 1; }
  cd "$repo" || return 1
  source "$activate" || return 1

  curl -fsS --max-time 5 http://127.0.0.1:8002/health >/dev/null 2>&1 && host_ok=1
  curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && coord_ok=1

  if [ "$host_ok" -eq 1 ] && [ "$coord_ok" -eq 1 ]; then
    echo 'Host and Coordinator are already healthy.'
    coord_health="$(curl -fsS http://127.0.0.1:8000/health)" || return 1
    printf '%s\n' "$coord_health" | python -m json.tool || return 1
    printf '%s\n' "$coord_health" | python -c '
import json, sys
h=json.load(sys.stdin)
if h.get("quorum_policy") != "majority":
    raise SystemExit("ERROR: Coordinator is not using majority quorum")
if str(h.get("minimum_trusted_client_quorum")) != "2":
    raise SystemExit("ERROR: Coordinator minimum trusted quorum is not 2")
if str(h.get("trusted_client_quorum_override")) != "1":
    raise SystemExit("ERROR: Coordinator is not in one-Client test mode; stop the current stack before switching modes")
print("One-Client quorum override is active.")
' || return 1
    return 0
  fi

  if [ "$host_ok" -eq 1 ] || [ "$coord_ok" -eq 1 ]; then
    echo 'ERROR: only one of Host/Coordinator is healthy. Inspect the partial stack before starting anything else.'
    ps -ef | grep -E \
      'run_host_stack.py|uvicorn.*host.main|uvicorn.*coordinator.main' \
      | grep -v grep || true
    return 1
  fi

  export COORDINATOR_MINIMUM_TRUSTED_CLIENT_QUORUM=2
  export COORDINATOR_TRUSTED_CLIENT_QUORUM_OVERRIDE=1
  mkdir -p /scratch/legalfedllm-test/logs || return 1

  nohup python scripts/run_host_stack.py \
    --env-file .env.host \
    >"$log" 2>&1 < /dev/null &

  echo $! > "$pidfile"
  echo "Started Host stack parent PID $(cat "$pidfile")"
  echo "Log: $log"

  for i in $(seq 1 180); do
    host_ok=0
    coord_ok=0
    curl -fsS --max-time 2 http://127.0.0.1:8002/health >/dev/null 2>&1 && host_ok=1
    curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && coord_ok=1
    if [ "$host_ok" -eq 1 ] && [ "$coord_ok" -eq 1 ]; then
      break
    fi
    if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
      echo 'ERROR: Host stack exited during startup'
      tail -n 120 "$log"
      return 1
    fi
    sleep 5
  done

  curl -fsS --max-time 5 http://127.0.0.1:8002/health || {
    echo 'ERROR: Host did not become healthy'
    tail -n 120 "$log"
    return 1
  }
  echo

  coord_health="$(curl -fsS --max-time 5 http://127.0.0.1:8000/health)" || {
    echo 'ERROR: Coordinator did not become healthy'
    tail -n 120 "$log"
    return 1
  }
  printf '%s\n' "$coord_health" | python -m json.tool || return 1
  printf '%s\n' "$coord_health" | python -c '
import json, sys
h=json.load(sys.stdin)
assert h.get("quorum_policy") == "majority", h
assert str(h.get("minimum_trusted_client_quorum")) == "2", h
assert str(h.get("trusted_client_quorum_override")) == "1", h
print("One-Client Host/Coordinator stack is ready.")
' || return 1
}

legalfed_host_start_single_client
```

### Stop the Host/Coordinator stack safely

Use this when the stack must be stopped or when switching between normal
two-Client mode and one-Client test mode. Do not use broad commands such as
`pkill python` or `pkill uvicorn`.

```bash
legalfed_host_stop() {
  local repo=/scratch/legalfedllm-test/work/LegalFedLLM
  local -a stack_pids
  local i

  test -d "$repo" || { echo "ERROR: $repo is missing"; return 1; }
  cd "$repo" || return 1

  mapfile -t stack_pids < <(
    ps -eo pid=,args= |
    awk '/[r]un_host_stack.py/ {print $1}'
  )

  printf 'LegalFedLLM Host stack parent PIDs: %s\n' "${stack_pids[*]:-none}"

  if [ "${#stack_pids[@]}" -eq 0 ]; then
    echo 'No run_host_stack.py parent is running.'
    return 0
  fi

  if [ "${#stack_pids[@]}" -ne 1 ]; then
    echo 'ERROR: multiple run_host_stack.py parents found; inspect them before stopping anything.'
    return 1
  fi

  kill "${stack_pids[0]}" || {
    echo 'ERROR: failed to send SIGTERM to the Host stack parent'
    return 1
  }

  for i in $(seq 1 15); do
    if ! kill -0 "${stack_pids[0]}" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  echo 'Remaining LegalFedLLM processes:'
  ps -ef | grep -E \
    'run_host_stack.py|uvicorn.*host.main|uvicorn.*coordinator.main' \
    | grep -v grep || true
}

legalfed_host_stop
```

If the stack is being used by an active round, inspect that round before stopping
the Host. A selected Client disconnect does not shrink the signed round quorum.

### Issue enrollment tokens

Enrollment tokens are single-use. An already-enrolled saved Client profile keeps
its persisted Ed25519 identity and does not need a new token on every restart.
Do not paste enrollment tokens into public logs, commits or documentation.

#### One new Client

Make sure the Host/Coordinator stack is healthy first, then run:

```bash
legalfed_host_issue_token() {
  local repo=/scratch/legalfedllm-test/work/LegalFedLLM
  local activate=/scratch/legalfedllm-test/.venv/bin/activate

  test -d "$repo" || { echo "ERROR: $repo is missing"; return 1; }
  test -f "$activate" || { echo 'ERROR: Host virtual environment is missing'; return 1; }
  cd "$repo" || return 1
  source "$activate" || return 1

  curl -fsS --max-time 5 http://127.0.0.1:8002/health >/dev/null || {
    echo 'ERROR: Host is not healthy'
    return 1
  }
  curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null || {
    echo 'ERROR: Coordinator is not healthy'
    return 1
  }

  python scripts/issue_enrollment_token.py \
    --env-file .env.host \
    --token-only || return 1
}

legalfed_host_issue_token
```

Copy the printed token directly into exactly one new Client profile. Successful
enrollment consumes it.

#### Two new Clients

Issue the two tokens separately so each Client receives a different single-use
token:

```bash
legalfed_host_issue_two_tokens() {
  local repo=/scratch/legalfedllm-test/work/LegalFedLLM
  local activate=/scratch/legalfedllm-test/.venv/bin/activate

  test -d "$repo" || { echo "ERROR: $repo is missing"; return 1; }
  test -f "$activate" || { echo 'ERROR: Host virtual environment is missing'; return 1; }
  cd "$repo" || return 1
  source "$activate" || return 1

  curl -fsS --max-time 5 http://127.0.0.1:8002/health >/dev/null || {
    echo 'ERROR: Host is not healthy'
    return 1
  }
  curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null || {
    echo 'ERROR: Coordinator is not healthy'
    return 1
  }

  echo '=== Token for Client 1 ==='
  python scripts/issue_enrollment_token.py \
    --env-file .env.host \
    --token-only || return 1

  echo
  echo '=== Token for Client 2 ==='
  python scripts/issue_enrollment_token.py \
    --env-file .env.host \
    --token-only || return 1
}

legalfed_host_issue_two_tokens
```

Use the first token for the first new profile and the second token for the second
new profile. Do not reuse either token.

### Create a one-Client round

A one-Client round requires the Coordinator to be running in the one-Client test
mode described above, with `trusted_client_quorum_override` reported as `1` by
`/health`.

The repository defines two round slots:

```text
client-1 = Qwen3 1.7B
client-2 = Granite 3.3 2B
```

Use one of the two complete command blocks below.

#### One Qwen Client

```bash
legalfed_host_create_qwen_round() {
  local repo=/scratch/legalfedllm-test/work/LegalFedLLM
  local activate=/scratch/legalfedllm-test/.venv/bin/activate
  local coord_health

  test -d "$repo" || { echo "ERROR: $repo is missing"; return 1; }
  test -f "$activate" || { echo 'ERROR: Host virtual environment is missing'; return 1; }
  cd "$repo" || return 1
  source "$activate" || return 1

  coord_health="$(curl -fsS --max-time 5 http://127.0.0.1:8000/health)" || {
    echo 'ERROR: Coordinator is not healthy'
    return 1
  }
  printf '%s\n' "$coord_health" | python -c '
import json, sys
h=json.load(sys.stdin)
if str(h.get("trusted_client_quorum_override")) != "1":
    raise SystemExit("ERROR: start the Host/Coordinator in one-Client test mode first")
' || return 1

  read -r -p 'Enrolled Qwen Client ID: ' QWEN_CLIENT_ID
  test -n "$QWEN_CLIENT_ID" || { echo 'ERROR: Qwen Client ID is required'; return 1; }
  export QWEN_CLIENT_ID

  ROUND_CLIENT_SLOTS="client-1" \
  COORDINATOR_MINIMUM_TRUSTED_CLIENT_QUORUM="2" \
  COORDINATOR_TRUSTED_CLIENT_QUORUM_OVERRIDE="1" \
  python scripts/create_remote_round.py --env-file .env.host || return 1
}

legalfed_host_create_qwen_round
```

#### One Granite Client

```bash
legalfed_host_create_granite_round() {
  local repo=/scratch/legalfedllm-test/work/LegalFedLLM
  local activate=/scratch/legalfedllm-test/.venv/bin/activate
  local coord_health

  test -d "$repo" || { echo "ERROR: $repo is missing"; return 1; }
  test -f "$activate" || { echo 'ERROR: Host virtual environment is missing'; return 1; }
  cd "$repo" || return 1
  source "$activate" || return 1

  coord_health="$(curl -fsS --max-time 5 http://127.0.0.1:8000/health)" || {
    echo 'ERROR: Coordinator is not healthy'
    return 1
  }
  printf '%s\n' "$coord_health" | python -c '
import json, sys
h=json.load(sys.stdin)
if str(h.get("trusted_client_quorum_override")) != "1":
    raise SystemExit("ERROR: start the Host/Coordinator in one-Client test mode first")
' || return 1

  read -r -p 'Enrolled Granite Client ID: ' GRANITE_CLIENT_ID
  test -n "$GRANITE_CLIENT_ID" || { echo 'ERROR: Granite Client ID is required'; return 1; }
  export GRANITE_CLIENT_ID

  ROUND_CLIENT_SLOTS="client-2" \
  COORDINATOR_MINIMUM_TRUSTED_CLIENT_QUORUM="2" \
  COORDINATOR_TRUSTED_CLIENT_QUORUM_OVERRIDE="1" \
  python scripts/create_remote_round.py --env-file .env.host || return 1
}

legalfed_host_create_granite_round
```

The one-Client override is test-only. Return the Host to normal two-Client mode
before running the normal federation topology.

### Create the normal two-Client Qwen + Granite round

The Coordinator must be running in normal two-Client mode: majority policy,
minimum trusted quorum `2`, and no override.

```bash
legalfed_host_create_two_client_round() {
  local repo=/scratch/legalfedllm-test/work/LegalFedLLM
  local activate=/scratch/legalfedllm-test/.venv/bin/activate
  local coord_health

  test -d "$repo" || { echo "ERROR: $repo is missing"; return 1; }
  test -f "$activate" || { echo 'ERROR: Host virtual environment is missing'; return 1; }
  cd "$repo" || return 1
  source "$activate" || return 1

  coord_health="$(curl -fsS --max-time 5 http://127.0.0.1:8000/health)" || {
    echo 'ERROR: Coordinator is not healthy'
    return 1
  }
  printf '%s\n' "$coord_health" | python -c '
import json, sys
h=json.load(sys.stdin)
if h.get("quorum_policy") != "majority":
    raise SystemExit("ERROR: Coordinator is not using majority quorum")
if str(h.get("minimum_trusted_client_quorum")) != "2":
    raise SystemExit("ERROR: Coordinator minimum trusted quorum is not 2")
if str(h.get("trusted_client_quorum_override")) not in {"none", "", "None"}:
    raise SystemExit("ERROR: stop the one-Client test stack and restart normal two-Client mode")
' || return 1

  read -r -p 'Enrolled Qwen Client ID: ' QWEN_CLIENT_ID
  read -r -p 'Enrolled Granite Client ID: ' GRANITE_CLIENT_ID

  test -n "$QWEN_CLIENT_ID" || { echo 'ERROR: Qwen Client ID is required'; return 1; }
  test -n "$GRANITE_CLIENT_ID" || { echo 'ERROR: Granite Client ID is required'; return 1; }

  export QWEN_CLIENT_ID GRANITE_CLIENT_ID

  ROUND_CLIENT_SLOTS="client-1,client-2" \
  COORDINATOR_MINIMUM_TRUSTED_CLIENT_QUORUM="2" \
  COORDINATOR_TRUSTED_CLIENT_QUORUM_OVERRIDE="" \
  python scripts/create_remote_round.py --env-file .env.host || return 1
}

legalfed_host_create_two_client_round
```

For the Mistral Nemo Host, the supported alignment assignments are:

```text
Qwen3 1.7B
    → dtw:qwen3-1.7b--mistral-nemo-instruct-2407-v1

Granite 3.3 2B
    → dtw:granite3.3-2b-client--mistral-nemo-instruct-2407-v1
```

Unknown or unsupported model pairs fail closed.

### Verify and monitor a round

`create_remote_round.py` prints the newly created `round_id`. Enter that value
when prompted below; this is runtime input, not a filesystem or deployment
placeholder.

```bash
legalfed_host_show_manifest() {
  read -r -p 'Round ID printed by create_remote_round.py: ' ROUND_ID
  test -n "$ROUND_ID" || { echo 'ERROR: Round ID is required'; return 1; }
  export ROUND_ID

  curl -fsS \
    "http://127.0.0.1:8000/v1/rounds/${ROUND_ID}/manifest" |
  python -c '
import json, sys
d=json.load(sys.stdin)
print("round_id:", d["round_id"])
print("selected_client_ids:", d["selected_client_ids"])
print("quorum:", d["trusted_client_quorum"])
print("alignment_profiles:")
for cid, profile in d["selected_client_alignment_profiles"].items():
    print(" ", cid, "->", profile)
print("submission_deadline:", d["submission_deadline"])
' || return 1
}

legalfed_host_show_manifest
```

Then inspect the current state:

```bash
legalfed_host_round_status() {
  test -n "${ROUND_ID:-}" || {
    echo 'ERROR: ROUND_ID is not set; run legalfed_host_show_manifest first or set ROUND_ID manually.'
    return 1
  }

  curl -fsS \
    "http://127.0.0.1:8000/v1/rounds/${ROUND_ID}/status" |
  python -c '
import json, sys
d=json.load(sys.stdin)
print("round_id:", d["round_id"])
print("state:", d["state"])
print("accepted_client_ids:", d["accepted_client_ids"])
print("accepted_count:", len(d["accepted_client_ids"]))
print("message:", d.get("message"))
' || return 1
}

legalfed_host_round_status
```

For continuous compact monitoring:

```bash
legalfed_host_watch_round() {
  test -n "${ROUND_ID:-}" || {
    echo 'ERROR: ROUND_ID is not set; run legalfed_host_show_manifest first or set ROUND_ID manually.'
    return 1
  }

  watch -n 5 "
curl -fsS http://127.0.0.1:8000/v1/rounds/${ROUND_ID}/status |
python -c '
import json,sys
d=json.load(sys.stdin)
print(\"state:\", d[\"state\"])
print(\"accepted:\", d[\"accepted_client_ids\"])
print(\"count:\", len(d[\"accepted_client_ids\"]))
print(\"message:\", d.get(\"message\"))
'
"
}

legalfed_host_watch_round
```

In a one-Client test round, quorum `1` permits the accepted Client package to
advance the round. In the normal two-Client topology, a new round begins as
`COLLECTING` with quorum `2`; after the first accepted package it remains
`COLLECTING` at `1/2`, and after the second trusted package reaches quorum the
Coordinator seals the accepted set and proceeds through Host integration and
distillation.

An expired collecting round is evaluated normally by its status endpoint. If it
passes its submission deadline without reaching trusted quorum, it becomes
`SKIPPED`. Terminal `SKIPPED`, `COMPLETED` and `ABORTED` rounds no longer block
creation of the next round. Do not hand-edit or delete generated round state.

### Normal Host workflow summary

For routine operation on this A40 installation:

```text
1. Enter /scratch/legalfedllm-test/work/LegalFedLLM and activate
   /scratch/legalfedllm-test/.venv.
2. Check the A40 with nvidia-smi.
3. Check Host :8002 and Coordinator :8000.
4. Start the appropriate stack mode only if both services are stopped:
   - normal two-Client mode for Qwen + Granite federation;
   - one-Client override mode only for controlled single-Client testing.
5. Verify Coordinator quorum mode from /health.
6. Issue one unique enrollment token for each genuinely new Client profile.
7. Create either the one-Client or two-Client round with the corresponding
   command block above.
8. Monitor the signed round through the Coordinator status endpoint.
9. Leave the detached Host stack running while the round is active unless there
   is an explicit reason to stop it.
```

## Development record

LegalFedLLM has progressed from deterministic protocol fixtures to real model
training, heterogeneous tokenizer alignment, Host validation/rollback, reverse
learning, portable desktop Clients, and a quorum-2 heterogeneous desktop round.
The table below records the implementation and experimental milestones that
have been exercised in the project.

| Capability | Proof-of-concept result |
| --- | --- |
| Deterministic protocol/mock path | Implemented and regression-tested |
| Canonical D^P/D^V dataset boundary | Implemented with stable IDs, ordering and semantic hashes |
| Signed manifests and Knowledge Package transport | Implemented with Ed25519, hashes, nonces, replay controls and bounded artifacts |
| Qwen3 1.7B real Client LoRA path | Implemented and real-tested, including a complete Host→Qwen reverse candidate acceptance path |
| Granite 3.3 2B real Client LoRA path | Implemented and real-tested; Granite knowledge has been accepted in a real heterogeneous multi-Client round |
| Qwen/Granite → Mistral Nemo DTW alignment | Both Client-specific Nemo alignment profiles exercised in real model federation |
| SafeFed-inspired package screening/trust | Implemented as a proof-of-concept defense-in-depth layer |
| Mistral Nemo real Host LoRA training | Implemented and real-tested on the proof-of-concept Host GPU |
| Hidden D^V Host validation | Implemented; candidates can be promoted or rejected while retaining the active adapter |
| Client reverse learning | Implemented with verified Host package intake, selective teaching, consent, stale-parent protection and held-out candidate validation |
| Submission acknowledgement reconciliation | Implemented and regression-tested |
| Portable desktop Client | Linux x86_64 AppImage and Windows x64 thin-launcher paths implemented and physically exercised |
| Constrained-GPU safeguards | Implemented across Client training, reference inference and reverse validation, with `gpu-memory.jsonl` diagnostics |
| LOCAL OpenAI-compatible inference | Implemented through the active Client Transformers/PEFT state with lazy resident-session reuse |
| HOST OpenAI-compatible forwarding | Implemented through the authenticated Client→Coordinator→Host boundary |
| Local-learning queue | Implemented; Constant Learning is enabled by default and explicit Learn/Dismiss remains available when disabled |
| Real multi-Client federation | Demonstrated with Windows Qwen + Linux Granite under normal trusted quorum 2; both Client packages were accepted and Host validation completed |
| Heterogeneous rollback behavior | Demonstrated: a completed multi-Client round can reject the candidate Host adapter and retain the previous accepted adapter |
| Formal DP-SGD / production PKI / production-calibrated poisoning defense | Intentionally not claimed by this research proof of concept |

Across the acceptance history, the complete bidirectional Qwen↔Mistral-Nemo
path and the true Qwen+Granite quorum-2 multi-Client path were both exercised.
Those results did not need to occur in one identical hardware run for the
README to record them as separate measured experiments; measured experiment
claims remain separate from architecture claims.

## Verified bidirectional single-Client baseline

A fresh unattended bidirectional run has been verified with:

- a local **Qwen/Qwen3-1.7B** Client;
- a remote **mistralai/Mistral-Nemo-Instruct-2407** Host;
- the Coordinator colocated with the remote NVIDIA GPU Host;
- one private-data Client epoch;
- five Host reference-data epochs;
- one reverse Client reference-data epoch;
- DTW alignment;
- DualMinCE teacher selection;
- top-k `4`;
- maximum sequence length `4096` with truncation rejected; and
- `chat_sft_answer_only_v1` supervision.

The verified run produced:

| Measurement | Result |
| --- | --- |
| Coordinator terminal state | `COMPLETED` |
| Client Knowledge Package submission | normal `201 Created` acknowledgement |
| Manual acknowledgement recovery | not required |
| Client D^P samples | 565 |
| Client stored token rows | 133,336 |
| Client Knowledge Artifact | 6,943,336 bytes |
| Coordinator post-alignment Client-package trust | accepted, trust score `0.33` |
| Forward Host-teacher selections | 565 / 565 |
| Forward Qwen-teacher selections | 0 / 565 |
| Host reference-data epochs | 5 |
| Host optimizer steps | 710 |
| Host adapter | 0 → 1, promoted |
| D^V macro mean answer-token CE | `2.4609236204 → 1.9685784675` |
| Required D^V improvement | `0.001` |
| Observed D^V improvement | `0.4923451529` |
| Reverse Host-teacher samples | 508 |
| Qwen training adapter | 1 → 2 |
| Reverse candidate decision | `candidate_accepted` |
| Client `last_completed_round` | `round-000001` |
| tmux evidence result | `PASS` |

The Qwen package was not rejected by the safety layer. It was accepted, aligned,
and eligible for selective distillation. However, the Host baseline had the
lower answer-token CE on every D^P sample, so DualMinCE selected the Host as the
forward teacher on all 565 samples.

That means the run verifies the full forward protocol, safety, tokenizer
alignment, selection, Host training, D^V validation, publication, automatic
sync, and reverse path. It does **not** show that Qwen knowledge caused the Host
validation improvement, because the Qwen Client was selected as teacher on
`0 / 565` forward samples.

The reverse direction is different: the promoted Host was selected as teacher
on 508 transfer samples, so the run contains substantive selected-teacher
**Host-to-Qwen** transfer.

The numerical values above describe one experiment. They are not protocol
constants or general model-performance claims.

## Architecture and protocol boundary

LegalFedLLM follows the FedMKT/FATE-LLM idea of transferring model behavior
instead of averaging heterogeneous model parameters.

A normal round is:

```text
1. Coordinator signs a round manifest.
2. Selected Clients verify the manifest and exact D^P identity/order.
3. The desktop Client optionally consumes user-approved queued local-learning examples and promotes the resulting local LoRA update. If the queue is empty, the current adapter is retained.
4. Each Client runs teacher-forced inference over D^P using its current model-native adapter.
5. Each Client signs and uploads a Knowledge Package + safetensors artifact.
6. Coordinator verifies transport, identity, replay, dataset and safety rules.
7. Eligible Client outputs are aligned into the Host tokenizer space.
8. DualMinCE chooses one teacher per D^P sample.
9. Host trains a model-native LoRA candidate from sparse targets.
10. Host evaluates active and candidate adapters on hidden D^V.
11. Host promotes or rolls back the candidate.
12. Host publishes a signed post-decision Knowledge Package over D^P.
13. Clients verify the Host package and execute Client-owned reverse alignment.
14. The desktop Client asks for user consent only when at least one Host-teacher sample exists.
15. After consent, the Client trains a local reverse candidate; without consent, the round is finalized locally without reverse training.
16. The held-out Client quality gate, stale-parent check and explicit consent govern local adoption.
17. Client commits the round only after the reverse decision completes.
```

The core architectural rule is:

```text
model-specific weights stay model-specific
knowledge crosses the federation boundary
```

There is no heterogeneous LoRA averaging step.

## Implemented components

The repository currently includes:

- signed round manifests and Knowledge Packages;
- Ed25519 service/Client identities;
- filesystem-backed round and checkpoint state;
- deterministic replay/nonce protection;
- canonical reference-dataset hashing and ordered sample identities;
- deterministic D^P/D^V splitting;
- a reviewed Greek Law Digest importer for the thesis dataset;
- real Transformers/PEFT Client and Host training backends;
- answer-only causal supervision;
- `safetensors` Knowledge Artifacts;
- bounded multipart package transport;
- exact tokenizer-artifact validation;
- demand-driven vocabulary mapping;
- FedMKT-compatible DTW token alignment;
- sparse top-k trainer targets instead of full-vocabulary dense targets;
- minimum-CE / DualMinCE teacher selection;
- SafeFed-inspired package plausibility and post-alignment trust analysis;
- trust-gated selective distillation;
- Host D^V validation and atomic promotion/rollback;
- immutable Client reverse-training jobs;
- Qwen reverse LoRA candidate training and adoption gates;
- exact accepted-submission receipt reconciliation;
- local Client Ollama installation/profile compatibility checks;
- direct Transformers + PEFT LOCAL serving for the active Client adapter;
- authenticated HOST inference forwarding through a bounded Coordinator queue;
- OpenAI-compatible `legalfedllm-local` and `legalfedllm-host` provider endpoints;
- portable multi-profile desktop state with per-profile Client identities;
- user-approved one-use local-learning queues and reverse-learning consent;
- Agent-owned OpenSSH tunnel supervision;
- PySide6 desktop source plus a lightweight Linux x86_64 AppImage path with a versioned Docker Client runtime; and
- split-machine tmux orchestration with preserved run evidence.

The deterministic mock path remains available for protocol and failure-policy
regression tests. Mock loss/safety values are fixtures; they are not real-model
measurements.

## Portable desktop Client

The repository contains a Linux x86_64 AppImage Client and a native Windows x64
thin-launcher Client. Both use the same PySide6 desktop/controller and portable
sibling `LegalFedLLM-data` state boundary, while their local-AI/runtime
integration differs by platform.

Linux AppImage mode keeps the GUI/controller on the host and runs the heavy
Client Transformers/PEFT runtime in a versioned Docker image. The GUI owns the
host OpenSSH tunnel and the Docker Client consumes that already-established
forward through an explicit external-tunnel contract.

Windows uses a thin `LegalFedLLM.exe` launcher with adjacent source/runtime files,
a release-local `.venv`, native Ollama and native AnythingLLM Desktop. The
experimental model placement uses Qwen3 1.7B on the constrained Windows GPU and
Granite 3.3 2B on the larger Linux Client GPU.

Both desktop paths have been physically exercised for profile/enrollment,
SSH/Agent startup, local-AI integration, real model participation and persistent
portable state. The quorum-2 heterogeneous GUI round accepted both Windows Qwen
and Linux Granite Knowledge Packages and proceeded through Host integration and
D^V validation.

The locked desktop behavior is:

- one Client codebase with Qwen and Granite selected through approved model profiles;
- PySide6 GUI, with PyInstaller as the per-OS executable bundler and an AppImage
  wrapper on Linux;
- portable state in a sibling `LegalFedLLM-data/` directory rather than an
  OS-global LegalFedLLM application-data directory;
- each saved profile owns an independent Client ID, Ed25519 identity, adapter
  state and one-time enrollment;
- the Client Agent is a child process of the GUI. OpenSSH asks for the SSH
  password in the launch terminal before the Agent API starts, and the password
  is never handled or stored by LegalFedLLM;
- AppImage mode keeps the host SSH tunnel outside the Client Docker container and
  gives the container an explicit external-tunnel contract;
- the global `Debug Mode` option is off by default, leaving only the launch/SSH
  terminal; when enabled it additionally opens the state and NVIDIA/GPU
  diagnostic terminals;
- Low VRAM Mode is on by default. Changing it persists the next-launch value but
  does not close the running desktop; the current Agent keeps its effective
  settings until the user manually restarts LegalFedLLM;
- CUDA Client execution uses micro-batch 1, preserving the configured effective
  batch through gradient accumulation, non-reentrant gradient checkpointing,
  SDPA attention, and 64-token checkpointed loss projection. Saved decoder
  activations are offloaded to pinned system RAM and restored for backward;
  model computation and LoRA optimization stay on GPU. This trades RAM and
  transfer/recomputation time for VRAM without changing signed sequence lengths;
- CUDA reference inference and reverse validation use batch 1 and at most
  64-token causal chunks. Low VRAM Mode uses 32-token inference chunks and,
  on Linux, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It no longer
  disables the baseline protections when off;
- Client GPU admission limits PyTorch's allocator, leaves display/driver
  headroom, and checks actual frozen-model bytes before transferring the model.
  A refused workload remains on its existing adapter; there is no automatic
  CPU fallback, truncation, or retry. The minimum workspace check is a lower
  bound, not a proof that all activations fit or that a platform reset is impossible;
- `gpu-memory.jsonl` under the Client data directory records phase markers,
  effective execution settings, and memory counters without prompts or answers.
  CUDA operations are asynchronous: these markers narrow a failure interval,
  rather than proving the exact GPU instruction that failed;
- Ollama models are installed by the user. LegalFedLLM checks that the selected
  profile's expected Ollama model is installed, but federated training and LOCAL
  serving use the exact pinned Transformers + PEFT state;
- before real-model participation, the Client resolves its signed Client↔Host
  alignment profile and verifies both exact pinned tokenizers from the local
  Hugging Face cache. This preflight does not download Host model weights;
- D^P is downloaded from the Coordinator and verified against the signed round
  manifest;
- LOCAL AnythingLLM interactions enter the generated per-profile `train.jsonl`
  queue automatically while `Constant Learning` is enabled (the default).
  Turning it off restores the explicit `Learn from this` / `Dismiss` decision;
- queued local examples are consumed once and removed only after the resulting
  adapter is safely promoted; failed training restores the queue;
- the main GUI action is `Participate in the current federated round`;
- reverse Host learning requires a verified Host package, at least one selected
  Host-teacher sample, and explicit user consent. A terminal first-receipt
  freshness rejection is persisted per profile/round so the GUI does not spam
  repeated stale-package dialogs; transient preview failures back off and remain
  retryable;
- the OpenAI-compatible provider exposes `legalfedllm-local` and
  `legalfedllm-host`. LOCAL Transformers serving keeps a lazy resident model
  session between prompts, releases it before exclusive training jobs, and
  reloads the active checkpoint on the next LOCAL request;
- on Linux/AppImage, the repository carries the `legalfed-ai/` Docker Ollama +
  AnythingLLM bundle. On profile activation the desktop seeds a writable copy
  under `LegalFedLLM-data/legalfed-ai/`, preserves compatible existing
  AnythingLLM configuration when migrating from `~/legalfed-ai`, and rewrites
  the Generic OpenAI provider to the active profile's loopback Client Agent;
- on Linux/AppImage, a fresh local-only profile starts its Client Agent and
  Docker local-AI services without OpenSSH. Enrolled profiles still use the
  managed SSH tunnel for federation connectivity. After the Client Agent becomes
  healthy, LegalFedLLM starts Ollama, verifies the required compatibility model
  without pulling it, starts AnythingLLM, and opens `http://127.0.0.1:3001/` in
  the host default browser;
- on Windows, LegalFedLLM uses native Ollama and AnythingLLM Desktop rather than
  Docker. It verifies the required Ollama model without pulling it, detects or
  launches AnythingLLM Desktop, and configures the single reserved Generic OpenAI
  connection through AnythingLLM's local backend API. For a fresh, not-yet-onboarded
  AnythingLLM installation, LegalFedLLM selects Generic OpenAI with
  `legalfedllm-local` and completes the one-time onboarding automatically. For an
  already-onboarded installation, its existing default provider is preserved until
  the user explicitly selects **Local** or **Host** from LegalFedLLM's Windows
  **AnythingLLM model** control. That explicit switch selects Generic OpenAI and
  the chosen LegalFedLLM model; Host selection requires an enrolled profile and a
  live Coordinator connection. The active profile's `legalfedllm-local` and
  `legalfedllm-host` models remain behind the authenticated loopback Client Agent;
- on GUI exit, the profile-specific AppImage Client Docker stack is brought down
  automatically. The user separately chooses whether managed
  Ollama/AnythingLLM services should stop or remain running; stopping them
  preserves their persistent volumes; and
- AnythingLLM native Generic OpenAI tool calling is disabled for the current 1.0
  path. Ordinary chat and RAG are supported; Agent/tool calling is not yet part
  of the LegalFedLLM OpenAI-compatibility contract.

The desktop build helpers are:

```bash
python scripts/build_desktop.py
python scripts/build_desktop.py --appimage   # Linux; requires appimagetool
```

The normal PyInstaller build and Linux AppImage are distinct packaging paths.
The Linux AppImage is built as a small `--onedir` controller bundle and stages
the Docker Client runtime inputs rather than collecting the complete ML runtime
inside the AppImage. Windows and Linux artifacts must be built on their
respective operating systems; PyInstaller is not a cross-compiler.

## Pinned model profiles

### Clients

| Profile | Model | Revision | Current role/status |
| --- | --- | --- | --- |
| `qwen3-1.7b-lora-v1` | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Real Client path verified, including Windows quorum-2 participation and a complete bidirectional Qwen↔Nemo baseline |
| `granite-3.3-2b-instruct-client-lora-v1` | `ibm-granite/granite-3.3-2b-instruct` | `652c333dc5066f2a1764854a1bcd0ce67163d74f` | Real Client path verified; Granite knowledge was accepted from the Linux desktop in the quorum-2 heterogeneous PoC round |

The Qwen profile uses `Qwen3ForCausalLM`, `Qwen2TokenizerFast`, vocabulary size
151,936 and the non-thinking Qwen chat-template mode. The Granite profile uses
`GraniteForCausalLM`, `GPT2TokenizerFast` and vocabulary size 49,159.

Both Client profiles use the current PoC LoRA contract:

```text
rank:             8
alpha:            16
dropout:          0.05
target modules:   q_proj, k_proj, v_proj, o_proj
bias:             none
task:             CAUSAL_LM
```

### Hosts

| Profile | Model | Revision | Current role/status |
| --- | --- | --- | --- |
| `mistral-nemo-instruct-2407-host-lora-v1` | `mistralai/Mistral-Nemo-Instruct-2407` | `04d8a90549d23fc6bd7f642064003592df51e9b3` | Current remote Host profile with real GPU training/validation verified |
| `granite-3.3-2b-instruct-host-lora-v1` | `ibm-granite/granite-3.3-2b-instruct` | `652c333dc5066f2a1764854a1bcd0ce67163d74f` | Retained compatibility Host profile |

The pinned Mistral Nemo Host uses `MistralForCausalLM`,
`PreTrainedTokenizerFast`, vocabulary size 131,072 and the same rank-8 LoRA
target modules. Its default test serving backend remains `mock`; the desktop HOST
inference path may explicitly select `transformers` so it serves the active Host
PEFT adapter directly. Ollama is not used to pretend a promoted PEFT adapter has
been published.

`compose.host-ml.yaml` remains the local Granite ML Compose profile. The Mistral
Nemo Host path runs directly from a Python virtual environment on the remote GPU
Host.

## Pinned alignment profiles

LegalFedLLM recognizes four exact bidirectional tokenizer-alignment contracts:

| Client | Host | Alignment ID | Status |
| --- | --- | --- | --- |
| Qwen3 1.7B | Granite 3.3 2B | `dtw:qwen3-1.7b--granite3.3-2b-v1` | Supported compatibility/validation profile |
| Granite 3.3 2B | Granite 3.3 2B | `dtw:granite3.3-2b-client--granite3.3-2b-host-v1` | Supported compatibility/validation profile |
| Qwen3 1.7B | Mistral Nemo | `dtw:qwen3-1.7b--mistral-nemo-instruct-2407-v1` | Real heterogeneous profile verified |
| Granite 3.3 2B | Mistral Nemo | `dtw:granite3.3-2b-client--mistral-nemo-instruct-2407-v1` | Real heterogeneous profile exercised in the quorum-2 desktop PoC round |

These contracts pin model/tokenizer revisions, tokenizer artifact hashes,
special-token state, vocabulary ranges, padding behavior, chat-template hashes,
and word-boundary rules. Unknown or mismatched profiles fail closed.

A signed live manifest now carries a deterministic Client-ID keyed alignment mapping.
The Coordinator derives each assignment from the selected Client's registered
model/tokenizer profile, the signed Host profile, and the approved alignment registry;
callers do not choose the mapping. Unknown pairs fail closed. Protocol `1.1` binds this
representation into the signed manifest. Model-free regression coverage verifies a
mixed Qwen + Granite → Mistral Nemo manifest, Client-specific package enforcement,
quorum sealing, duplicate rejection, canonical manifest hashing/signing, and Host
training-input construction with an independent tokenizer/alignment contract for each
accepted Client. Real Qwen + Granite model execution has been exercised in the quorum-2 desktop PoC round.

## Repository layout

```text
LegalFedLLM/
├── client/                     Client API, training, package generation,
│                               reverse training and adoption validation
├── coordinator/                round lifecycle, quorum, package intake,
│                               safety/trust, Host integration and persistence
├── host/                       Host API, pinned profiles, real PEFT training,
│                               validation and post-decision publication
├── shared/                     protocol, crypto, datasets, artifacts,
│   └── fedmkt_core/            FedMKT parity/adaptation, alignment, selection,
│       └── ml/                 optional upstream-derived ML components
├── config/
│   ├── container.env.example   remote Host/Coordinator template
│   └── clients.env.example     split Client template
├── scripts/
│   ├── bootstrap.py            role-aware private environment bootstrap
│   ├── create_remote_round.py  Host-side split-round creation
│   ├── run_host_stack.py       direct Host + Coordinator process launcher
│   ├── run_remote_round.py     Client register/train/participate/sync driver
│   ├── run_split_round_tmux.sh split-machine orchestration/evidence dashboard
│   ├── validate_fedmkt_alignment.py
│   ├── measure_knowledge_packages.py
│   └── demo_round.py
├── tests/                      model-free and opt-in real-model regression tests
├── tools/datasets/             source-specific GLD inspection/import tooling
├── compose.yaml                local development stack
├── compose.clients.yaml        role-separated Qwen/Granite Client stack
├── compose.host-ml.yaml        local Granite Host ML override
├── requirements.txt
├── requirements-desktop.txt
└── THIRD_PARTY_NOTICES.md
```

Generated/runtime state is intentionally outside the tracked source boundary.
`.gitignore` excludes `.env*`, `data/`, `artifacts/`, logs, PEM/private-key files,
virtual environments, and local delivery archives.

## Clean-clone and private-state contract

A clean checkout contains source code, model/profile contracts, Compose files,
environment templates, tests, and bootstrap tooling. It does not contain:

- deployment secrets;
- Ed25519 private keys;
- private Client training examples;
- downloaded model weights;
- trained adapters;
- runtime round state;
- the copyrighted Greek Law Digest PDF; or
- generated thesis D^P/D^V datasets.

Create role-specific private environment files with `scripts/bootstrap.py`.
The command creates the file only when absent and reuses it rather than silently
overwriting it.

Host/Coordinator example:

```bash
python scripts/bootstrap.py host \
  --output .env.host \
  --runtime-root /path/to/legalfedllm-runtime
```

After the Host/Coordinator is running, issue one single-use Client enrollment token:

```bash
python scripts/issue_enrollment_token.py \
  --env-file .env.host
```

Give that token to exactly one new Client, then bootstrap the Client:

```bash
python scripts/bootstrap.py client \
  --output .env.remote-client \
  --registration-token '<single-use enrollment token>'
```

The Client generates its own Ed25519 identity under its private runtime directory.
Successful enrollment binds that public key to the Client record and consumes the token.
The token is not used for later rounds. D^P downloads and accepted-submission receipt
lookups authenticate with signed Client requests using the persisted Ed25519 identity.

For controlled one-Client proof-of-concept testing only, the Host bootstrap
supports:

```bash
python scripts/bootstrap.py host \
  --output .env.host \
  --runtime-root /path/to/legalfedllm-runtime \
  --trusted-quorum-override 1
```

Do not use the one-Client override as the normal federation configuration.

### Source bootstrap, enrollment and D^P delivery

The source tree is sufficient to create the private runtime credentials needed by
LegalFedLLM; real secrets are not expected to be committed. Host bootstrap
generates Host administrative/internal tokens. Client bootstrap requires one
Host-issued enrollment token and generates the Client-local administrative
secrets. The desktop path similarly generates a per-profile Client admin token
and persists its Ed25519 identity under that profile's private state.

The GLD corpus itself is not generated from GitHub source and is not embedded in
the public release. The authorized Host/Coordinator must already have the real
D^P and hidden D^V files at its configured dataset paths. When a registered
Client is selected for a signed round, the Client automatically makes a signed
request to:

```text
GET /v1/rounds/{round_id}/reference-dataset
```

The Coordinator authorizes the enrolled Client, serves the round-bound D^P
snapshot, and the Client verifies its dataset ID, semantic hash and ordered
sample IDs before caching it locally. D^V has no Client download endpoint and
remains on the Host/Coordinator side.

Therefore a source Client connected to an already-provisioned Host/Coordinator
can enroll, generate its own local secrets/identity state, and obtain D^P through
the protocol without receiving the copyrighted GLD source PDF or Host-only D^V.

## Private Client training data

Real Client training reads local UTF-8 JSONL. Each record has the logical form:

```json
{
  "schema_version": "1.0",
  "example_id": "private-example-001",
  "prompt": "A private local instruction or question.",
  "answer": "The private local target answer."
}
```

The Client enforces unique/non-empty IDs, non-empty prompt/answer strings,
deterministic order and hashing, a supported schema, and answer-only
supervision. Overlength examples fail rather than being silently truncated.

For the role-separated Client Compose stack, the configured private directory is
mounted read-only and the training file is expected as `/private/train.jsonl`.
The raw private examples are never written to Coordinator storage or included in
a Knowledge Package.

## Reference dataset boundary

A canonical reference record contains:

```json
{
  "schema_version": 1,
  "dataset_id": "example-reference",
  "dataset_version": "v1",
  "sample_id": "example-ch001-s001-q001",
  "chapter": "Example Chapter",
  "section": "Example Section",
  "question": "What is the question?",
  "gold_answer": "The reference target answer.",
  "source": {
    "document_id": "example-document",
    "page_start": 10,
    "page_end": 11
  }
}
```

The authoritative runtime representation is UTF-8 JSONL with one sample per
line. The semantic hash binds the schema, dataset identity, and ordered
`sample_id`, `chapter`, `section`, `question`, and `gold_answer` values. Source
page metadata is provenance and is not part of the semantic hash.

The generic split groups samples by `(chapter, section)` and preserves source
order. A one-sample section belongs entirely to D^P. Otherwise D^P receives the
first `floor(0.8 * n)` samples and D^V receives the remainder. Source-specific
GLD follow-up grouping is resolved before the generic split.

### Thesis GLD identities

The reviewed thesis dataset is derived from a pinned 713-page 2012 Greek Law
Digest source copy.

| Dataset | Samples | Semantic SHA-256 |
| --- | ---: | --- |
| Complete canonical corpus | 738 | `cf5c81dcecaab58848c1afb0e99f86bcf5fd32823c2aaee34a65f8a3dc21d49` |
| D^P shared reference set | 565 | `5d855a429d43b70eb146aeb11cda1f675c05d6465bea0792796fdcd8d6ceb231` |
| D^V hidden validation set | 173 | `1e40a74799b9900ff8b9a9e05dd379fd0c00226370625f7da1fdca13142b83b5` |

D^P is distributed only to selected Clients for a signed round. D^V has no
public Client download endpoint and remains on the Coordinator/Host side for
candidate validation.

The Greek Law Digest source is private/copyrighted. Neither the source PDF nor
the generated thesis dataset belongs in the public repository.

### Offline GLD tooling

Source-specific inspection/import code remains outside the generic dataset
boundary under `tools/datasets/`. The workflow is deliberately review-oriented:
ambiguous structure should produce deterministic warnings or blocking review
rather than silent guessing.

Generated files should be regenerated from importer rules rather than manually
edited. Provenance, source identity, sample count, and semantic hashes must stay
reproducible.

## Prompt and answer-only supervision

Client and Host training use the shared supervision label:

```text
chat_sft_answer_only_v1
```

Prompt rendering is model/tokenizer-specific, but loss is computed only over the
answer portion. The same answer-token boundary is used when producing CE
evidence for teacher selection and validation.

Overlength training/reference rows are rejected when the active profile uses the
`reject` truncation policy.

## Knowledge Package and Artifact contract

A Knowledge Package is a signed JSON envelope plus a bounded `safetensors`
artifact. The manifest binds at least:

- round identity;
- sender identity and public-key trust context;
- model and tokenizer profile;
- adapter identity;
- exact Client-specific alignment assignment for Client packages;
- reference-dataset identity and ordered sample set;
- artifact byte size and SHA-256;
- top-k setting;
- nonce/replay state; and
- signature.

The artifact stores sparse knowledge rather than full dense vocabulary logits.
For each answer-token row it retains top-k token IDs/logits plus evidence needed
for exact CE checks and alignment.

Coordinator validation checks tensor names, dtypes, shapes, offsets,
finite-valued data, vocabulary bounds, top-k consistency, sample offsets,
artifact size/hash, and package/manifest identities before the package can be
accepted.

## FedMKT alignment and teacher selection

Different tokenizers cannot compare token IDs directly. LegalFedLLM therefore
maps tokenizer pieces into a shared word/character representation and applies a
DTW-based alignment adapted from FedMKT/FATE-LLM.

The implementation keeps the alignment deterministic and cacheable:

```text
Client sparse logits
      ↓
Client token strings / word-boundary mapping
      ↓
DTW alignment against Host tokenization
      ↓
aligned sparse teacher evidence
      ↓
Host-vs-teacher answer-token CE comparison
      ↓
DualMinCE teacher choice per sample
```

The Client does not automatically become a teacher merely because its package
was accepted. Safety eligibility and selective CE comparison are separate
conditions. If the Host is better on a sample, the Host remains the teacher for
that sample.

## Safety model

LegalFedLLM adapts the defense-in-depth idea of Safe-FedLLM to behavioral
Knowledge Packages rather than treating heterogeneous Client parameters as
aggregatable updates.

The safety path includes:

```text
structural validation
      ↓
pre-alignment plausibility checks
      ↓
post-alignment disagreement analysis
      ↓
trust score / eligibility
      ↓
trust-gated teacher selection
      ↓
hidden D^V Host outcome validation
      ↓
promotion or rollback
```

Pre-alignment checks are intentionally cheap and deterministic. They catch
malformed/non-finite artifacts, invalid token IDs/order, and gross pathological
concentration/repetition without assuming that natural Qwen/Nemo vocabulary
heterogeneity is malicious.

Repeated-frequency calculations use linear-time counting rather than repeated
full-list scans. Artifact loading and CPU-heavy pre-alignment inspection run off
the Coordinator event-loop thread so health/status/receipt endpoints remain
responsive while safety inspection is in progress. Safety still gates durable
acceptance; the Coordinator does not report an accepted receipt before the
package has passed the required checks and been persisted.

Post-alignment checks compare behavior after heterogeneous token spaces have
been normalized enough for meaningful disagreement analysis. Trust can gate or
down-weight selective distillation.

Hidden D^V validation is an additional outcome barrier, not a complete poisoning
proof. Targeted/backdoor behavior outside D^V coverage can still evade aggregate
validation metrics, so the safety layers are complementary rather than
interchangeable.

## Host training and D^V promotion

After package acceptance and alignment, the Coordinator constructs sparse Host
trainer inputs. The Host trains a model-native LoRA candidate; no Client LoRA
weights are inserted into the Host.

The Host then evaluates the active adapter and candidate on hidden D^V using the
same answer-only loss contract. The decision is fail-closed:

```text
candidate improves by required margin → promote atomically
candidate does not improve            → keep active adapter
validation/training failure            → do not promote
```

The post-decision Host Knowledge Package is generated from the active adapter
after that decision, so Clients sync from the actual promoted/retained Host
state.

## Client reverse distillation and adoption

After Host publication, the Client:

1. downloads and verifies the signed Host package;
2. checks exact round/reference identities and its signed Client-specific alignment assignment;
3. aligns Host sparse knowledge into the Client tokenizer space;
4. selects Host-teacher samples using the reverse CE rule;
5. creates an immutable reverse-training job;
6. trains a model-native Client LoRA candidate when transfer samples exist;
7. evaluates the candidate against the held-out Client quality gate; and
8. commits or rejects the candidate before marking the round complete.

Reverse training is Client-owned. The Coordinator/Host never installs a Client
adapter. Client-side reverse-candidate adoption does not use a LoRA safety probe;
promotion is governed by the held-out quality gate, stale-parent protection, forced
validation rejection controls used by tests, and explicit user consent. Coordinator-
side Knowledge Package safety/trust screening remains separate and unchanged.

A real Transformers Client may legitimately participate before it has ever
created a PEFT checkpoint. In that base-only state, the accepted round snapshot binds
the Client package to the frozen base-model state hash with adapter version 0 and no
checkpoint hash. If reverse learning is later approved, the Client creates a fresh
transient LoRA on the exact pinned base model, verifies that its initial effective
LoRA delta is zero, trains the first candidate, validates the plain base model against
the candidate, and persists a PEFT checkpoint only if the learned candidate is
promoted. A rejected first candidate leaves the Client base-only.

The same lifecycle applies to the first approved local-learning batch: a fresh
zero-effect LoRA is only a transient training parent. It is not promoted as a durable
adapter merely to satisfy PEFT. The first durable Client checkpoint is the learned
candidate itself; if that training operation fails, the Client remains base-only.

The desktop marks a Host-package preview complete only after a successful preview; an
interrupted or cache-related preview remains retryable for the same completed round.
Host-package timestamp freshness is enforced when the package is first received and
verified. After that successful preview, later user consent may consume only the same
immutable cached package/reverse job; wall-clock freshness is not re-applied as a
human decision deadline.

## Submission acknowledgement and retry semantics

Knowledge submission is transactional from the Client's point of view. The
Client keeps the exact pending package, artifact, and adapter snapshot until it
has authoritative evidence that the exact package was accepted.

If the POST acknowledgement is lost or ambiguous, the Client can query:

```text
GET /v1/rounds/{round_id}/submissions/{client_id}/receipt
```

The receipt lookup is authenticated with the enrolled Client's Ed25519 identity and
must match the exact round, Client, and package hash. Each protected request is
bound to the HTTP method/path and includes a timestamp plus nonce; stale, forged,
or replayed requests are rejected. The Client commits only when the exact package
is confirmed accepted. Wrong hash/client/round or an unaccepted submission fails
closed and leaves the pending package intact.

This recovery path handles lost acknowledgements without turning retries into a
second logical submission.

## tmux split-round orchestration

`scripts/run_split_round_tmux.sh` orchestrates the role-separated real test while
keeping the remote Host/Coordinator and local Client visibly separate.

It creates panes/windows for:

```text
remote Host/Coordinator + remote GPU monitor
round runner / automatic round creation
local Client logs + local GPU monitor
Coordinator state / final evidence collector
SSH tunnel
```

The script uses an SSH ControlMaster so the user enters the remote password once.
After the remote Coordinator becomes healthy, the script asks it to issue one
single-use enrollment token for the fresh local Client and passes that token only to
the Client startup/registration path. It no longer requires a persistent shared
registration secret in the Host environment. The tunnel is used for Coordinator
traffic; the remote Host remains bound to loopback.

The generated cleanup script stops the tmux session, Client runtime, and SSH
ControlMaster while preserving run evidence.

### Final evidence behavior

Runner exit does not immediately freeze Coordinator evidence. The finalizer:

```text
records runner exit
      ↓
continues observing persisted Coordinator state
      ↓
COMPLETED / SKIPPED / ABORTED
      or bounded evidence timeout
      ↓
writes coordinator-final-state.json
      ↓
captures final Client health
      ↓
computes PASS / FAIL
```

Observation is evidence collection only. It does not retry a submission, call
`/sync`, mutate Client state, or otherwise recover a failed run automatically.

## Running a split Client manually

The local Client uses `compose.clients.yaml` and a role-specific environment
file containing the one-time enrollment token for its first registration. A typical
workflow is:

```bash
docker compose \
  -p legalfedllm-split \
  --env-file .env.remote-client \
  -f compose.clients.yaml \
  --profile qwen \
  up -d --build
```

Then run the round driver from the repository environment:

```bash
python scripts/run_remote_round.py
```

For unattended testing, prefer `scripts/run_split_round_tmux.sh` so startup,
state observation, GPU monitoring, final evidence, and cleanup are collected
consistently.

## Tests and verification

### Ordinary repository suite

From the repository root, run the complete ordinary suite. On Linux, clear any
inherited AppImage marker so source-mode browser tests do not accidentally take
the AppImage `xdg-open` branch:

```bash
env -u APPIMAGE python -m unittest discover -v
```

On Windows PowerShell:

```powershell
python -m unittest discover -v
```

Some real-model tests are opt-in and require the pinned model/tokenizer artifacts
plus the expected CUDA/Transformers/PEFT environment. A missing optional ML
dependency should be distinguished from a source-code regression.

### Desktop/AppImage packaging and lifecycle regressions

Focused model-free coverage for the lightweight AppImage/Docker Client boundary
is:

```bash
python -m unittest -v \
  tests.test_appimage_packaging \
  tests.test_desktop_client_docker
```

This covers the AppImage packaging boundary, graphical terminal relaunch,
sanitized host-browser launch, the versioned Client Docker runtime, portable
profile/model mounts, writable local-learning state, the external SSH-tunnel
contract, and full Docker-stack cleanup when the Client Agent terminates.

### Reliability and safety regressions

Focused model-free coverage includes:

```bash
python -m unittest -v \
  tests.test_enrollment_auth \
  tests.test_package_safety \
  tests.test_submission_reconciliation \
  tests.test_split_round_tmux
```

These tests cover, among other things:

- single-use enrollment-token issuance/consumption and restart persistence;
- signed Client request authentication, stale/forged/replay rejection;
- persistent Client identity/enrollment without token reuse;
- lost-after-acceptance acknowledgement reconciliation;
- exact hash/client/round receipt matching;
- fail-closed unconfirmed submissions;
- responsive receipt/status handling while safety validation is still running;
- no premature accepted receipt during validation;
- linear-time safety frequency calculations; and
- terminal evidence collection after runner exit.

Shell syntax should also be checked with:

```bash
bash -n scripts/run_split_round_tmux.sh
```

### Real Qwen acceptance

Real Qwen tests are opt-in and require the configured tokenizer/model artifacts
and CUDA environment. They validate private Client LoRA training, real D^P
Knowledge Package generation, and package/reverse paths.

### Reverse Qwen acceptance

The reverse real-model path verifies that a Host-derived sparse training job can
train a new Qwen PEFT candidate and exercise the independent local adoption
logic. These tests qualify the implemented Client lifecycle; they do not qualify
a production malicious-update classifier.

### Full-D^P heterogeneous alignment validation

The deterministic alignment runner can be executed with:

```bash
python scripts/validate_fedmkt_alignment.py \
  --reference data/derived/gld2012/reference.jsonl \
  --output artifacts/fedmkt-alignment-validation.json \
  --mapping-cache artifacts/fedmkt-alignment-cache \
  --identity-dir artifacts/fedmkt-validation-identities \
  --maximum-sequence-length 4096 \
  --top-k 4
```

It records deterministic mapping/alignment identities and selection behavior;
it is not a model-quality benchmark.

## Service APIs

### Coordinator — port 8000

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Coordinator status/quorum policy |
| `GET` | `/v1/identity` | Coordinator + Host public identity |
| `POST` | `/v1/enrollment-tokens` | Admin issue one single-use Client enrollment token |
| `POST` | `/v1/clients/register` | Consume enrollment token and bind Client profile/public key |
| `POST` | `/v1/rounds` | Create/sign a round manifest |
| `GET` | `/v1/rounds/current` | Retrieve current manifest |
| `GET` | `/v1/rounds/{id}/manifest` | Retrieve one manifest |
| `GET` | `/v1/rounds/{id}/status` | Retrieve round state |
| `GET` | `/v1/rounds/{id}/reference-dataset` | Selected-Client D^P download |
| `POST` | `/v1/rounds/{id}/knowledge` | Upload signed package + artifact |
| `GET` | `/v1/rounds/{id}/submissions/{client_id}/receipt` | Authenticated exact accepted-submission receipt |
| `GET` | `/v1/rounds/{id}/safety` | Admin safety reports for the round |
| `GET` | `/v1/rounds/{id}/host-knowledge` | Download signed post-decision Host package |
| `POST` | `/v1/generate` | Proxy Host generation |

D^P download and receipt lookup require signed Client request authentication using
the public key bound at enrollment. The enrollment token is accepted only by the
registration endpoint and is consumed after successful registration. The
safety-report and enrollment-token issuance endpoints are admin-protected.

### Client — loopback port 8001

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Client state/backend status |
| `POST` | `/v1/register` | Register with Coordinator |
| `POST` | `/v1/local-train` | Local training outside a round |
| `POST` | `/v1/rounds/{id}/local-train` | Train against one signed round |
| `POST` | `/v1/participate` | Legacy current-round participation |
| `POST` | `/v1/rounds/{id}/participate` | Generate and submit exact round package |
| `POST` | `/v1/rounds/{id}/sync` | Verify Host package and execute reverse decision |
| `POST` | `/v1/generate` | Local mock/Ollama generation |
| `GET` | `/v1/ollama/models` | List configured Ollama models |
| `POST` | `/v1/ollama/inspect` | Inspect an Ollama model |

Client administrative endpoints require `X-Client-Admin-Token`.

### Host — private/loopback port 8002

The Host internal API is protected by `X-Internal-Token`. It exposes internal
identity, reference-data loading, reference knowledge, sparse training-job
intake, candidate training, candidate validation, post-decision knowledge, and
generation. In the remote topology it is loopback-only and is contacted by the
colocated Coordinator.

## Persistence and idempotence

LegalFedLLM does not require a database. Each role owns a filesystem state tree.
Important persisted Client state includes:

```text
identity and active adapter state
round-bound training records
PEFT checkpoints
verified D^P caches
pending and accepted Knowledge Packages
submission receipts
adapter snapshots
Host package cache
immutable reverse jobs and sparse artifacts
candidate/validation/adoption records
```

Important Coordinator state includes:

```text
hashed single-use enrollment-token records
registered Client identities
persisted signed-request nonce records
signed manifests
accepted submissions
safety reports and trust history
round state
Host baseline package
integration audit
Host training job/receipt/result/validation decision
post-decision Host package
round audit events
```

Retries are exact where possible. A pending Client submission is revalidated
against its round, model profile, training record, checkpoint, and artifact.
Once accepted, the package/artifact/snapshot set is immutable.

## Security and privacy boundary

Implemented security controls include:

- one-time admin-issued Client enrollment tokens stored only as hashes;
- persistent Ed25519 Client identities and signatures after enrollment;
- method/path/timestamp/nonce-bound signed Client requests with replay rejection;
- canonical JSON hashing/signing;
- exact artifact byte-size and SHA-256 binding;
- registered Client public keys;
- signed Coordinator manifests;
- model/tokenizer/adapter/reference-dataset/sample-order binding;
- nonce and package-hash replay protection;
- selected-Client authorization for D^P download;
- bounded multipart/package sizes;
- strict tensor names/dtypes/shapes/offsets/finite-value validation;
- gold-token/log-normalizer evidence consistency checks;
- temporary-file cleanup and immutable accepted storage;
- pre- and post-alignment package safety reports;
- trust-gated selection;
- hidden D^V validation and Host rollback;
- round/checkpoint provenance; and
- append-only audit records.

These controls do **not** provide confidentiality by themselves. Ed25519 proves
origin/integrity; it does not encrypt HTTP traffic or stored artifacts.

The current DP report enforces protocol/policy consistency only. Real model
training is ordinary LoRA training, not DP-SGD, and LegalFedLLM makes no formal
differential-privacy claim.

Knowledge/logit sharing can itself reveal information. The project should not be
described as formally private merely because raw Client examples and Client LoRA
weights remain local.

## Ollama boundary

Ollama is an optional **serving** boundary, not the federated training runtime.
Real training uses Transformers/PEFT.

The Qwen Client can serve through local Ollama (`qwen3:1.7b`). The Granite
compatibility profile can use `granite3.3:2b`. The pinned Mistral Nemo Host
currently supports `mock` serving only in the LegalFedLLM profile.

Promoting a PEFT training adapter does not currently export/import it into an
Ollama model automatically. Consequently `training_adapter_version` can advance
while `serving_adapter_version` remains unchanged. That is a known deployment
boundary and should not be interpreted as evidence that reverse training failed.

## FedMKT upstream/adaptation record

The optional machine-learning components under `shared/fedmkt_core/ml/` were
extracted and adapted from `FederatedAI/FATE-LLM`, package
`fate_llm.algo.fedmkt`, at commit:

```text
0c63377e468f0f62a9bdf5fb32424688b9478553
```

LegalFedLLM retains the reviewed DTW/minimum-CE behavior while replacing FATE
communication/orchestration with its own protocol, HTTP, persistence, and
security layers. It also uses answer-only supervision, deterministic
demand-driven vocabulary mapping, and operational sparse targets.

See:

- `shared/fedmkt_core/UPSTREAM.md`
- `shared/fedmkt_core/PARITY.md`
- `shared/fedmkt_core/LICENSE`
- `THIRD_PARTY_NOTICES.md`

The upstream FATE Context, Guest/Host/Arbiter channels, FATE-Flow, and parameter
aggregation wrappers are not part of LegalFedLLM.

## Accurate project claim

A defensible project summary is:

> LegalFedLLM implements a research proof of concept for heterogeneous federated
> language-model learning in which model-native LoRA weights and raw
> private Client examples remain local while signed behavioral Knowledge Packages
> move over a common reference dataset. The implementation includes real Qwen and
> Granite Client LoRA paths, Mistral Nemo Host LoRA training, signed and replay-
> protected federation, Client-specific DTW tokenizer alignment, SafeFed-inspired
> package screening/trust, DualMinCE teacher selection, hidden D^V Host validation
> with promotion/rollback, signed Host publication, Client-owned reverse
> distillation/adoption, portable Linux and Windows desktop Clients, and a real
> quorum-2 Windows-Qwen + Linux-Granite heterogeneous GUI round.

The earlier bidirectional Qwen↔Mistral-Nemo baseline demonstrated successful
reverse candidate adoption. The later multi-Client round demonstrated real
heterogeneous quorum, both Client Knowledge Packages being accepted, and a valid
Host candidate rejection/rollback outcome. These are measured experimental results rather than claims of production
security, formal differential privacy, or general model-quality improvement.
