import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data

class Trainer():
    def __init__(self,data):
        self.pipeline = self.set_pipeline()
        self.y = data["fare_amount"]
        self.X = data.drop("fare_amount", axis=1)
        self.X_train,self.X_test,self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.15)
        self.run()
        
        self.y_pred = self.pipeline.predict(self.X_test)

    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                            ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                            ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                        remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                        ('linear_model', LinearRegression())])
        return pipe
    
    def run(self):
        self.pipeline.fit(self.X_train, self.y_train)
        

    def evaluate(self):
        rmse = np.sqrt(np.mean((self.y_test - self.y_pred) ** 2))
        return rmse
    
trainer = Trainer(get_data())
print(trainer.evaluate())