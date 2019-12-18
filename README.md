# IBM AI Enterprise Workflow Certification - Capstone Solution Files


Usage notes
===========

All commands are from this directory.

To test app.py
--------------

``` {.bash}
~$ python app.py
```

or to start the flask app in debug mode

``` {.bash}
~$ python app.py -d
```

Go to <http://0.0.0.0:8080/> and you will see a basic website that can
be customtized for a project.

To test the model directly
--------------------------

see the code at the bottom of [model.py]{.title-ref}

``` {.bash}
~$ python model.py
```

To build the docker container
-----------------------------

``` {.bash}
~$ docker build -t iris-ml .
```

Check that the image is there.

``` {.bash}
~$ docker image ls
```

You may notice images that you no longer use. You may delete them with

``` {.bash}
~$ docker image rm IMAGE_ID_OR_NAME
```

And every once and a while if you want clean up you can

``` {.bash}
~$ docker system prune
```

Run the unittests
-----------------

Before running the unit tests launch the [app.py]{.title-ref}.

To run only the api tests

``` {.bash}
~$ python unittests/ApiTests.py
```

To run only the model tests

``` {.bash}
~$ python unittests/ModelTests.py
```

To run all of the tests

``` {.bash}
~$ python run-tests.py
```

Run the container to test that it is working
--------------------------------------------

``` {.bash}
~$ docker run -p 4000:8080 iris-ml
```

Go to <http://0.0.0.0:4000/> and you will see a basic website that can
be customtized for a project.
