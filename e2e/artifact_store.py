import os
import io
import logging
import sys

from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class ArtifactStore:
    def __init__(self, project_name: str, bucket_name: str):
        self.gcs_project_name = project_name
        self.gcs_bucket_name = bucket_name

    def _get_storage_path(self, flow_name: str, run_id: str, file_name: str) -> str:
        return f"{flow_name}/{run_id}/{file_name}"

    def _get_flow_storage_directory(self, flow_name: str, run_id: str, remote_subdir: str) -> str:
        #ToDo: use os.path.join instead
        return f"{flow_name}/{run_id}/{remote_subdir}"

    def store(self, data: bytes, storage_path: str) -> str:
        from google.cloud import storage

        client = storage.Client(project=self.gcs_project_name)
        bucket = client.get_bucket(self.gcs_bucket_name)

        blob = bucket.blob(storage_path)

        with io.BytesIO(data) as f:
            # TODO: Catch exceptions and report back.

            # Google recommends setting `if_generation_match=0` if the
            # object is expected to be new. We don't expect collisions,
            # so setting this to 0 seems good.
            blob.upload_from_file(f, if_generation_match=0)
            logging.info(f"The model is stored at {storage_path}")

    def fetch(self, remote_path: str, local_path: str) -> str:
        from google.cloud import storage

        client = storage.Client(project=self.gcs_project_name)
        bucket = client.get_bucket(self.gcs_bucket_name)

        blob = bucket.blob(remote_path)

        # Create any directory that's needed.
        p = Path(local_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(local_path)

    def fetch_directory(self, remote_directory: str, local_directory: str) -> str:
        """Download all blobs from a directory within a gcs bucket to a directory on local machine.

        The filename of each blob once downloaded to the `local_directory` is derived from the blob name
        within the `remote_directory` (i.e. relative blob name wrt to the `remote_directory` parameter).
        This means, the full path of the each downloaded blob on local machine can be computed by replacing
        `remote_directory` prefix in the blob name with `local_directory`.

        Directories will be created automatically as needed to accommodate blob
        names that include slashes."""

        from google.cloud import storage
        from google.cloud.storage import transfer_manager

        client = storage.Client(project=self.gcs_project_name)
        bucket = client.get_bucket(self.gcs_bucket_name)

        blob_names_in_remote_directory = client.list_blobs(self.gcs_bucket_name, prefix=remote_directory, delimiter=None)
        blob_file_pairs = [(blob, local_directory + blob.name.removeprefix(remote_directory)) for blob in blob_names_in_remote_directory]

        # Create any directory that's needed.
        p = Path(local_directory)
        p.mkdir(parents=True, exist_ok=True)

        results = transfer_manager.download_many(blob_file_pairs, skip_if_exists=False)
        for blob_file_pair, result in zip(blob_file_pairs, results):
            # The results list is either `None` or an exception for each blob in the input list, in order.
            if isinstance(result, Exception):
                raise Exception(f"Failed to download blob: '{blob_file_pair[0].name}' to location: '{blob_file_pair[1]}' due to exception: '{result}'")
            else:
                logger.info(f"Downloaded blob: '{blob_file_pair[0].name}' to location: '{blob_file_pair[1]}'")

    def store_flow_data(self, data: bytes, filename: str) -> str:
        from metaflow import current

        deployment_path = self._get_storage_path(
            current.flow_name, current.run_id, filename
        )

        self.store(data, deployment_path)

        return deployment_path

    def fetch_flow_data(self, flow_name: str, run_id: str, file_name: str) -> str:
        from google.cloud import storage

        path = self._get_storage_path(
            flow_name=flow_name, run_id=run_id, file_name=file_name
        )

        self.fetch(remote_path=path, local_path=path)

        return path

    def fetch_all_flow_data_from_directory(self, flow_name: str, run_id: str, flow_subdir: str, local_dir: str) -> str:
        storage_dir = self._get_flow_storage_directory(
            flow_name=flow_name, run_id=run_id, remote_subdir=flow_subdir
        )

        self.fetch_directory(remote_directory=storage_dir, local_directory=local_dir)

        return local_dir

if __name__ == '__main__':
    artifact_store: ArtifactStore = ArtifactStore("moz-fx-mlops-inference-nonprod", "mf-models-test1")
    #artifact_store.fetch_all_flow_data_from_directory("CodeAutocompletionFlow", "107", "trained/checkpoint-2", "./models/checkpoint-500")
    artifact_store.fetch_directory("TrainingFlowBQ/37", "./test-model/checkpoint")
