ML model deployment
===========================

deploy and inference a machine learning model (built on a car dataset) using Docker and FastAPI.

1. With terminal navigate to the root of this repository ./ML
--------------------------------------------------------

2. Build docker image
---------------------
   code-block::

    docker build -t image_name -f A1_Rregression/app/Dockerfile .

3. Run container
----------------
   code-block::

    docker run --name container_name -p 8000:8000 image_name

4. Output will contain
----------------------
INFO:     Uvicorn running on http://0.0.0.0:8000

Use this url in chrome to see the model frontend;
use http://0.0.0.0:8000/docs for testing the model in the web interface.

5. Query model
--------------
 #. easy way:

            go to "http://0.0.0.0:8000/"  fill the form
            with the data it's asking you. 
            example of how to fill the form: car year: 2020
                                            engine: 7.24
                                            max_power: 70
                                            transmission: 1
                                            owner: 1
                                            km_driven: 11.512925
                                            fuel: 0