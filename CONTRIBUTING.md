# Contributing

1. Create a feature branch.
2. Add tests for any public API changes.
3. Run:

```bash
pytest -q
ruff check .
```

4. Open a pull request with a concise description of the modeling or software change.

When adding new robot models, prefer the existing 13-state convention:

```text
x = [p_wb, q_wxyz, v_b, omega_b]
```

and expose a SciPy-compatible method:

```python
dx = model.state_derivative(t, x, input_or_gait)
```
