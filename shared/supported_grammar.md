# Supported Grammar – Sprint 1

## EBNF

```
equation     ::= expression '=' expression
expression   ::= term (('+' | '-') term)*
term         ::= factor (('*' | '/') factor)*
factor       ::= NUMBER
               | VARIABLE
               | NUMBER VARIABLE          /* implicit multiply */
               | NUMBER '(' expression ')'  /* implicit multiply */
               | '(' expression ')'
               | '-' factor               /* unary minus */
NUMBER       ::= [0-9]+
VARIABLE     ::= 'x'
```

## Implicit multiplication rules

| Written | Normalised |
|---|---|
| `2x` | `2 * x` |
| `2(x+1)` | `2 * (x + 1)` |

## Constraints

- Exactly one `=` sign per equation
- At most one variable (`x`)
- No exponents or functions in Sprint 1
