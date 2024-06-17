import numpy as np
import typer

from maturin_burn_test import hello, pyadd

app = typer.Typer()

@app.command()
def hi(name: str):
    typer.echo(f"Hello {name}")
    typer.echo(hello())


@app.command()
def calc(a: int, b: int, gpu: bool = False):
    x = np.linspace(0, a, 10, dtype=np.float32)
    y = np.linspace(0, b, 10, dtype=np.float32)
    z = pyadd(gpu, x, y)
    typer.echo(z)
