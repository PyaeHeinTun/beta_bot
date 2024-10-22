# BETA BOT

BetaBot is a crypto market maker bot that buy and sell assets based on machine learning.
This bot only use numpy , pandas and scikit-learn, so performance of betabot is good for large scale computing.

# Bot Photo

![alt text](https://i.imgur.com/jotXP0V.png)
![alt text](https://i.imgur.com/K0gFjFv.png)
![alt text](https://i.imgur.com/65wPmf5.png)

## Table of Contents

- [Installation](#installation)
- [Features](#features)

## Installation

### Software requirements

- [Python >= 3.12.3](http://docs.python-guide.org/en/latest/starting/installation/)
- [pip](https://pip.pypa.io/en/stable/installing/)
- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html) (Recommended)

### Minimum hardware requirements

To run this bot you should have a cloud instance with a minimum of:

- Minimal system requirements: 1GB RAM, 1GB disk space, 1vCPU
- Python Version 3.12.3
- Ubuntu OS

### Installation

- python3 -m venv .venv (Create New Python Virtual Environment)
- source .venv/bin/activate
- chmod +x ./setup.sh & ./setup.sh
- pip3 install -r requirements.txt (Install Python lib)
- in config.json change ["exchange"]["real_key"] , ["exchange"]["real_secret"] with your exchange api_key
- in config.json change ["exchange"]["pair_whitelist"] to desired assets.
- in config.json change ["telegram"]["token"] , ["telegram"]["user_id"] , ["telegram"]["chat_id"] with your telegram account.
- python3 main.py (Start The Bot)

## Features

- BetaBot will fetch automatically and trained every 60 minutes.
- Every Minutes new candles is appeared, it will predict new candles directions.
- After predictions, system automatically places limit orders according to predicted results.
