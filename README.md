# 🎯 Recs, Bandits & Reinforcement Learning

🚀 A hands-on Python project implementing modern recommendation techniques, multi-armed bandits, reinforcement learning agents and learning-to-rank models. Built with ❤️ for research, prototyping and practical learning.

---

## 📚 Features

✅ Recommender Systems:
- 📌 Content-Based Filtering  
- 🤝 Collaborative Filtering  
- 🔀 Hybrid Recommenders (CB + CF)

✅ Bandits:
- 🎲 Epsilon-Greedy  
- 📈 Upper Confidence Bound (UCB)  
- 🎯 Thompson Sampling  

✅ Reinforcement Learning (RL):
- 🧠 REINFORCE (Policy Gradient)  
- 📘 Q-Learning

✅ Learning to Rank (LTR):
- 🔹 Pointwise
- 🔸 Pairwise (RankNet)
- 🔻 Listwise (ListNet)

---

## ⚙️ Getting Started

Clone e instale as dependências com [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/orenatobr/recs_rl_bandits.git
cd recs_rl_bandits
poetry install --no-root
```

---

## 🚀 Example Usage

Execute o script principal:

```bash
poetry run python main.py
```

---

## 💂 Folder Structure

```
recs_rl_bandits/
├── recommender/        # Content-based, collaborative & hybrid recommenders
├── bandits/            # MAB agents: epsilon-greedy, UCB, TS
├── rl/                 # RL agents: REINFORCE, Q-learning
├── ltr/                # Learning-to-Rank: pointwise, pairwise, listwise
├── utils/              # Metrics, helpers
├── tests/              # Unit tests for all modules
├── main.py             # Entry point with example runs
├── pyproject.toml      # Poetry project file
└── .github/workflows/  # CI/CD with pre-commit, tests, release
```

---

## ✅ Requirements

- Python ≥ 3.11.8
- `numpy`, `scikit-learn`, `torch`
- Dev tools: `black`, `isort`, `flake8`, `pytest`, `pre-commit`

Instaladas automaticamente via `poetry install`

---

## 🧪 Run Tests

```bash
poetry run pytest
```

Ou com pre-commit:

```bash
pre-commit run --all-files
```

---

## 🕡 Flashcards (AnkiPro)

Quer aprender sobre sistemas de recomendação, RL e bandits no Anki?

[![Visit Website](https://img.shields.io/badge/Open-Click%20Here-blue)](https://ankipro.net/shared_deck/v2_Hgo1Ev4b5S_4961509)

---

## 🤝 Contribuindo

Sinta-se à vontade para abrir issues, pull requests ou sugestões!  
Este projeto é mantido por [Renato F Pereira](mailto:orenatobr@icloud.com) ☕

---

## 📦 Publicação no PyPI

Este projeto está pronto para publicação com **Poetry + Trusted Publishing**.

Ao criar uma nova release, o pacote é automaticamente publicado em:  
🔗 https://pypi.org/project/recs-rl-bandits/

---

