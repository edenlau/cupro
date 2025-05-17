Getting started
===============

Installation
------------
Create and activate a virtual environment, then install the Python dependencies:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

Running the server
------------------
Start the FastAPI application locally with:

.. code-block:: bash

   python src/main_web.py

Visit ``http://localhost:8081`` in your browser once the server is running.

Docker usage
------------
A ``Dockerfile`` is provided for containerising the application. Build and run with:

.. code-block:: bash

   docker build -t cupro-app .
   docker run --env-file env.example -p 8081:8081 cupro-app

AWS deployment
--------------
Push the built image to Amazon ECR and run it on ECS or another container service. The basic steps are:

.. code-block:: bash

   aws ecr create-repository --repository-name cupro-app
   aws ecr get-login-password | docker login --username AWS --password-stdin <aws-account-id>.dkr.ecr.<region>.amazonaws.com
   docker tag cupro-app:latest <aws-account-id>.dkr.ecr.<region>.amazonaws.com/cupro-app:latest
   docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/cupro-app:latest

