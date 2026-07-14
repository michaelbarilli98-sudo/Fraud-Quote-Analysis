# Fraud-Quote-Analysis

A personal Python project that analyses insurance quotation data to identify behavioural patterns and potential fraud indicators.

The tool uses Pandas to organise quote activity, compare changes made during customer sessions and flag records that may require further review.

## Features

- Tracks changes to customer information across quote sessions
- Identifies changes to occupation and annual mileage
- Detects premium reductions following quote changes
- Reviews repeated quote activity within short periods
- Produces rule-based fraud indicators for manual review

## Technologies

- Python
- Pandas

## Data

This project uses synthetic data only.

It does not contain employer data, customer information, internal fraud rules or proprietary systems.

## Limitations

The tool does not determine whether fraud has occurred.

Its outputs are indicators intended to support further investigation and human review.

## Running the project

Install the required package:

```bash
pip install pandas
