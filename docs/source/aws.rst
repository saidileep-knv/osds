NYC-TLC Trip Record Data Usecase: Amazon S3
===========================================

.. raw:: html

    <style> .red {color:#D0312D;  font-size:14px} </style>
    <style> .red {color:#D0312D;  font-size:14px} </style>

.. role:: indexred
.. role:: red

.. toctree::
   :maxdepth: 2
   
All of the capabilities of :red:`ObjectDataStorage` can also be applied to datasets that reside in serverless object storage services like S3, Google Cloud Storage, and Azure. 

Install the ``osds`` package to retrieve data using ``ObjectStorageDataset`` class.

.. testcode::

	!pip install osds

Here `NYC-TLC Trip Record Data <https://registry.opendata.aws/nyc-tlc-trip-records-pds/>`_, a large public dataset (consisting of more than 100M records of trips taken by taxis and for-hire vehicles in New York City) hosted on Amazon S3 is used to illustrate how a large dataset is imported from cloud services using OSDS.

A brief decription of the variables belonging to the Yello Taxi Trip Records in 2017 is provided below:

.. csv-table:: 
	:header: "Variable	", "Name		", "Type", "Description"
	:widths: 40, 50, 15, 200
	
	"Features (X)", "id", "character", "A unique identifier for each trip."
	"Features (X)", "vendor_id", "integer", "A code indicating the provider associated with the trip record. There appears to be 2 taxi companies."
	"Features (X)", "pickup_datetime", "character", "The date and time when the meter was engaged. This is currently a combination of date and time."
	"Features (X)", "dropoff_datetime", "character", "The date and time when the meter was disengaged. This is a combination of date and time."
	"Features (X)", "passenger_count", "integer", "The number of passengers in the vehicle (driver entered value). This is a count from upto 9."
	"Features (X)", "pickup_longitude", "numeric", "The longitude where the meter was engaged. These are geographical coordinates."
	"Features (X)", "pickup_latitude", "numeric", "The latitude where the meter was engaged."
	"Features (X)", "dropoff_longitude", "numeric", "The longitude where the meter was disengaged."
	"Features (X)", "dropoff_latitude", "numeric", "The latitude where the meter was disengaged."
	"Features (X)", "store_and_fwd_flag", "character", "This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server. Y=store and forward; N=not a store and forward trip."
	"Target (y)", "fare_amount", "integer", "Taxi fare at the end of the trip."
	
To access datasets from Amazon S3 Cloud Storage using ``ObjectStorageDataset`` in an efficient way, we specify the URL-style glob named parameter of the OSDS class starting with ``f"s3://``.

`eager_load_batches` is set to False to avoid the out-of-memory conditions. Usually, `eager_load_batches` is set to False for datasets that do not fit in the node and cluster memory.

Since, a public dataset is used for illustration, `'anon'` in the `storage_options` is set to True. This enables an unauthenticated access to the object storage.

.. testcode::

    from osds.utils import ObjectStorageDataset
    from torch.utils.data import DataLoader


    train_ds = ObjectStorageDataset(glob=f"s3://nyc-tlc/trip data/yellow_tripdata_2017-0*.csv",  
                                       storage_options = {'anon' : True }, 
                                       batch_size = 16384, 
                                       eager_load_batches=False)

    train_dl = DataLoader(train_ds, batch_size=None)

After ObjectStorageDataset is instantiated with a batch_size of 16,384 in the Python runtime of a compute node, the implementation of ObjectStorageDataset triggers a network transfer of nearly 600 dataset partitions (number_of_rows(9,700,000)/batch_size(16384)) from the S3 bucket to the file system cache of the compute node.

We now create a ``LinearRegressionModel`` class using ``torch.nn.Module``, and will train a regression model on the dataset.

.. testcode::

	import torch
	class LinearRegressionModel(torch.nn.Module):
		def __init__(self, input_size, output_size):
			super(LinearRegressionModel, self).__init__()
			self.linear = torch.nn.Linear(input_size, output_size)
		
		def forward(self, X):
			return self.linear(X)

Now instantiate the **LinearRegressionModel** class, and the optimizer to train the model.

.. testcode::

	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	model = LinearRegressionModel(input_size=7, output_size=1).to(device)
	LEARNING_RATE = 3e-02
	optimizer = torch.optim.SGD(model_parameters(), lr=LEARNING_RATE)
	
The ITERATION_COUNT is set to 300, and the model is trained on nearly 4.9M records (batch_size*ITERATION_COUNT) of the 116M records collected over the year. 

A GRADIENT_NORM of 0.1 is used to avoid gradient explosion while training the model, and the code for training a LinearRegressionModel is provided below:

.. testcode::
	
	GRADIENT_NORM = 0.1

	ITERATION_COUNT = 300

	losses = []
	for iter_idx, batch in zip(range(ITERATION_COUNT), train_dl):
		y_batch, X_batch = batch[:,-1].to(device), batch[:, 3:10].to(device)

		y_est = model(X_batch.float())
		mse = pt.mean((y_est - y_batch) ** 2)
		mae = pt.mean((y_est - y_batch))
		mse.backward()

		pt.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM) if GRADIENT_NORM else None

		optimizer.step()
		optimizer.zero_grad()

  		losses.append(mse.data.item())
  		if (iter_idx % 25 == 0):
    			print(f"Iteration: {iter_idx}, MSE: {round(mse.data.item(),4)}, MAE: {round(mae.data.item(), 4)} W: {model.weight.data.squeeze()}")
	
As the NYC-TLC Trip Records Data is a public dataset, we used the unauthenticated access to the object storage, but proper authentication needs to be ensured for data privacy. The sample notebook for building a basic linear regression model on the data extracted using OSDS can be fond `here <https://colab.research.google.com/drive/1c0KkJEDbIqqZLDmcz5VeVf4CqUGypITR#scrollTo=uAZ3fOJ3AvPw>`_.






