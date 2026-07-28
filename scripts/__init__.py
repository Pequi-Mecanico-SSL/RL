"""Utilitarios do projeto.

Os entrypoints importam diretamente os componentes de treino ou inferencia que
usam. Manter este modulo sem imports eager evita exigir Ray no deploy standalone.
"""