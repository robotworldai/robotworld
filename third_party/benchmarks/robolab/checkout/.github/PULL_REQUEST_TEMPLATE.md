# Description

<!-- What does this PR do? For bug fixes, describe the problem and how to reproduce it. -->

## PR type

<!-- Check one. See CONTRIBUTING.md for what we accept in each category. -->

- [ ] New asset (object / scene / task / robot / variation)
- [ ] Bug fix
- [ ] New policy backend (`policies/<name>/`) — requires a paper/project reference and prior contact with the maintainers (see CONTRIBUTING.md)
- [ ] Documentation
- [ ] Other (describe above)

## Checklist

- [ ] All commits are signed off (`git commit -s`) — see [DCO](https://github.com/NVlabs/RoboLab/blob/main/CONTRIBUTING.md#developer-certificate-of-origin-dco)
- [ ] New Python files carry the standard SPDX header (`Apache-2.0`, NVIDIA copyright — copy from any existing file)
- [ ] New assets are self-contained in their own folder, include their license, and are listed in `THIRD_PARTY_NOTICES.md`
- [ ] Binary files (`.usd`, meshes, textures, images) are tracked with git-LFS
- [ ] `uv run pytest tests/` passes locally
- [ ] New robots: env configs compile against the benchmark tasks (see `docs/robots.md`)

## Asset attribution

<!-- For asset contributions: citation, contact information, and affiliation so we can credit your work. -->
