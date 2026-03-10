# Delivery Customer Support Assistant (NLP + RL)

## Overview

This project develops an AI-powered customer support assistant designed for delivery-related services. The system uses Natural Language Processing (NLP) to classify customer intents from delivery inquiries and applies Reinforcement Learning (RL) to optimize response selection for customer support interactions.

The assistant aims to help automate responses to common delivery issues such as order tracking, delivery delays, address updates, and missing packages.

## Task

The main task of this project is **intent classification and response selection** for delivery-related customer queries. The system analyzes a user's message, determines the intent of the request, and selects the most appropriate response using a learned policy.

## Minimum Viable Product (MVP)

The initial system will implement and compare the following intent classification models:

* **Naive Bayes**
* **Support Vector Machine (SVM)**
* **BERT**

The predicted intent will then trigger a predefined rule-based response. Reinforcement learning or bandit-based policies will be explored to improve response selection over time.

Example intents may include:

* Order status inquiry
* Delivery delay report
* Address change request
* Missing package report

## Evaluation Metrics

The system will be evaluated using the following metrics:

* **Intent Classification F1 Score**
* **Precision and Recall**
* **Task Success Rate**
* **Cumulative Reward** for the reinforcement learning response policy

## Ethical Considerations

This project considers several ethical risks related to automated customer support systems. User messages may contain sensitive information, so training data will be anonymized and personal identifiers will not be stored. The NLP models may also misclassify user intents, which will be mitigated by using diverse datasets and evaluating model performance with standard metrics. Additionally, automated responses may provide incorrect information; therefore, the system will use controlled rule-based replies and escalate complex or uncertain cases to human agents when necessary.

## Setup

The project setup is **currently incomplete** as development has not started yet. Environment configuration, data download scripts, and execution instructions will be added in later milestones.

## Version

v0.1 – Initial repository setup and project proposal.
