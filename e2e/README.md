# End to end example

## Requirements for Setup:
1. Python version >= 3.10 
2. For running flows _non-locally_, you'll need Weights and Biases and Outerbounds Metaflow accounts configured. For that, you'll need the help of an MLOps admin. Please reach out to us via the #mlops channel in Slack, and we can help you get those accounts set up. You _can_ go ahead and get the flows running _locally_ without this step.
3. You'll need to [set up Application Default Credentials with your user credentials](https://cloud.google.com/bigquery/docs/authentication/getting-started#python).

## Recommendations for Setup:
1. A virtual environment to install the dependencies and follow the workflows in. You can use `venv` for this, which ships with Python.

## Train a model

1. From the e2e directory of this repository, run `python3.10 -m pip install -r requirements.txt` (or substitute in `python -m` if Python 3.10 is your default distribution).
2. Try to run the training flow locally with `python training-flow.py --metadata=local --environment=pypi run --offline True` 
3. To record the data in Weights and Biases, the `WANDB_API_KEY`, `WANDB_ENTITY` and `WANDB_PROJECT` environment variables need to be set. You can set these locally in your virtual envuronment via the command line, or if you're using CI, you can set them on CI. Then you can run this command: `python training-flow.py --environment=pypi run --with kubernetes`.
5. (_Optional_) To run the training on a cluster without recording informations to W&B, use the following command: `python training-flow.py --environment=pypi run --offline True --with kubernetes`

You can track the training progress on the Outerbounds UI.

## Stand up an example inference server
This flow enables deploying and running 2 applications locally.

1. From the e2e directory, run `pip install -r requirements.txt`
2. Run the applications locally:
    1. To run the first application (inference by the model trained in ## Train a model step) locally, run
    ```
    serve run forecast:app_builder flow-name=TrainingFlowBQ namespace=<MODEL NAMESPACE>
    ```
    where `<MODEL NAMESPACE>` is the namespace used to store the model in Metaflow, e.g. `user:aplacitelli@mozilla.com`.

    2. To run the second application (a simple multiplier) locally, run
    ```
    serve run multiply:app_builder factor=<VALUE>
    ```
    where `<VALUE>` should be set to a float value.

> [!NOTE]
> The previous steps follow Ray Serve [Local Development with HTTP requests](https://docs.ray.io/en/latest/serve/advanced-guides/dev-workflow.html#local-development-with-http-requests) workflow, allowing fast
> local iteration to prototype the inference server. The next steps are useful to test
> a deployment process similar to the one happening in production, but not strictly
> required for local development.

3. Autogenerate the config file

    To workaround a bug in the generator, please uncomment the last lines referring to `app` in both `forecast.py` and `multiply.py` (we
    need to reference `app` and not `app_builder` otherwise generator would complain with `TypeError: Expected 'forecast:app_builder' to
    be an Application but got <class 'function'>.`) and run:
    ```
    serve build forecast:app multiply:app -o serve_config.yaml
    ```

4. Tweak the generated config `serve_config.yaml`, especially the `applications` section. Please set a unique `route_prefix` for each application.
    Here's a sample tweaked:

```yaml
# This file was generated using the `serve build` command on Ray v2.9.3.
proxy_location: EveryNode

http_options:
  host: 0.0.0.0
  port: 8000

grpc_options:
  port: 9000
  grpc_servicer_functions: []

logging_config:
  encoding: TEXT
  log_level: INFO
  logs_dir: null
  enable_access_log: true

applications:
- name: app1
  route_prefix: /forecast
  import_path: forecast:app_builder
  args:
    namespace: "user:CHANGEME" # TODO, change this!
    flow-name: TrainingFlowBQ
  runtime_env:
    pip:
      - outerbounds[gcp]
      - scikit-learn==1.3.1
  deployments:
  - name: Forecaster

- name: app2
  route_prefix: /multiply
  import_path: multiply:app_builder
  args:
    factor: 4
  runtime_env: {}
  deployments:
  - name: Multiplier
```

5. (_Optional_) Start a local ray cluster: `ray start --head`.
6. Deploy the server: `serve deploy serve_config.yaml`.
7. (_Optional_) Check the status via `serve status`.

## Testing the model
Try the first application: [`curl http://127.0.0.1:8000/forecast?q=1.4`](http://127.0.0.1:8000/forecast?q=1.4)

Try the second application: [`curl http://127.0.0.1:8000/multiply?q=1.8`](http://127.0.0.1:8000/multiply?q=1.8)

You can also visit these URLs in your browser.
