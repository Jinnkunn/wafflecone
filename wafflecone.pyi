def version():
    """Print the version of the package."""

def calculator(
    path: str,
    subspace_seeds: list[dict[str, list[str]]],
    exclude_words: list[str] = None,  # words to exclude from tokens
    user_friendly: bool = None,
    pca_dimension: int = None,
    model_name: str = None,
) -> "Calculator":
    """Print the calculator."""

def new_subspace_seeds(name: str, seeds: list[str]) -> "SubspaceSeed":
    """Create a new subspace seed."""

def visualize(port: int):
    """Visualize the calculator with web interface."""
