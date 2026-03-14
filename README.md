# Delivery Customer Support Assistant (NLP + RL)

## Development Team
Bermudo, Jeanne Clarrise T.

Magat, Maria Josephine M.

Pineda, Mary Alexa Ysabelle V.

Rebusa, Amber Kaia J.

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

### Risk of Incorrect Intent Classification
Although the model achieved high validation accuracy, it may still incorrectly classify user messages. Misclassification may cause the chatbot to return irrelevant responses and delay issue resolution.

To reduce this risk, the system should include fallback responses and escalation to a human support agent when the model confidence is low.

### Limitations of Automated Customer Support
Automated systems may fail to fully understand complex or ambiguous requests. Users expecting accurate assistance may become frustrated if the system cannot capture the full context of their problem.

Therefore, the chatbot should complement human customer support rather than replace it.

### Dataset Bias and Representativeness
The dataset used for training contains templated or synthetic queries. These queries may not reflect the diversity of real customer language.

Future work should include collecting more diverse datasets or incorporating real-world customer interactions.

### Privacy and Sensitive Information
Customer support messages may contain sensitive information such as order numbers, addresses, or personal identifiers. Improper handling of this data may lead to privacy concerns.

The system should avoid storing sensitive data unnecessarily and should anonymize stored conversation logs whenever possible.

## Setup

The project setup is **currently incomplete** as development has not started yet. Environment configuration, data download scripts, and execution instructions will be added in later milestones.

## Version

v0.1 – Initial repository setup and project proposal.
