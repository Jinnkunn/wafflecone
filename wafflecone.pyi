def version():
    """Print the version of the package."""

def calculator(
    path: str,
    subspace_seeds: list[dict[str, list[str]]],
    random_token_num: int = None,  # number of random tokens
    random_token_seed: int = None,  # random seed
    subspace_folder_path: str = None,  # folder path to save subspaces
    exclude_words: list[str] = None,  # words to exclude from random tokens
    user_friendly: bool = None,
    pca_dimension: int = None,
) -> "Calculator":
    """Print the calculator."""

def new_subspace_seeds(name: str, seeds: list[str]) -> "SubspaceSeed":
    """Create a new subspace seed."""

def visualize(port: int):
    """Visualize the calculator with web interface."""
