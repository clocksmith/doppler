# Git Version Control Fundamentals

Git is a distributed version control system designed to handle projects of any scale with speed and data integrity.

## Core Concepts
- **Repository**: The complete history and object database storing blobs, trees, commits, and tags.
- **Commit**: An immutable snapshot of the repository state identified by a cryptographic SHA hash.
- **Branches**: Movable lightweight pointers to commits. The default branch is typically named `main`.
- **Working Tree vs Staging Index**: The working tree contains files modified locally on disk. The staging area (`git add`) prepares changes into the next proposed snapshot.

## Important Operations
- `git commit -m "message"`: Persists staged changes into an immutable new commit node.
- `git merge`: Joins two development histories together, creating a merge commit when histories have diverged.
- `git rebase`: Replays local commits on top of another base tip to maintain a linear project history.
- `git checkout --detach`: Moves HEAD directly to a commit rather than following a branch name.
