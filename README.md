travel_entities
==============================

discover entities in travel discussions

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

License
-------

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Disclaimer

Summaries and analyses in this project are generated using OpenAI's models and may contain errors or inaccuracies. Please verify any generated content before relying on it.

## Running tests

This project uses [pytest](https://pytest.org) for unit tests. Run them with
`pytest` if it is available. If `pytest` is not installed, the tests can also be
executed with Python's built-in `unittest` runner:

```bash
python -m unittest discover -s tests -v
```

## Running the application

Install the dependencies from `requirements.txt` and start the FastAPI server:

```bash
python src/main_web.py
```

The service listens on port `8081` by default. Set the `PORT` environment variable to override it.

## Docker

Build the Docker image and run it locally using the provided `Dockerfile`:

```bash
docker build -t cupro .
docker run --env-file env.example -p 8081:8081 cupro
```

This starts the application inside a container and exposes it on <http://localhost:8081>.

## Amazon App Runner

To deploy with Amazon App Runner, the service expects an `apprunner.yaml` file in the repository root. This file defines how the application is built and run. Configure all required environment variables using the keys from [`env.example`](env.example) in the App Runner service settings.
