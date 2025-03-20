import unittest
from dataclasses import dataclass

import numpy as np
import pandas as pd

from booking.model_train_pipeline import ModelTrainPipeline


class ModelTrainPipelineTest(unittest.TestCase):

    def test_create_train_pipeline_booking(self):
        @dataclass
        class Booking:
            Booking_ID: str
            number_of_adults: int
            number_of_children: int
            number_of_weekend_nights: int
            number_of_week_nights: int
            type_of_meal: str
            car_parking_space: int
            room_type: str
            lead_time: int
            market_segment_type: str
            repeated: int
            P_C: int
            P_not_C: int
            average_price: float
            special_requests: int
            date_of_reservation: str

        # Create sample booking data
        X = pd.DataFrame(data=[
            Booking("B001", 2, 0, 1, 3, "Breakfast", 1, "Suite", 30, "Corporate", 0, 1, 0, 200.0, 0, "2021-07-01"),
            Booking("B002", 1, 1, 0, 2, "No Meal", 0, "Standard", 15, "Online", 1, 0, 1, 150.0, 1, "2021-07-02"),
            Booking("B003", 2, 2, 2, 4, "Breakfast", 1, "Deluxe", 45, "Direct", 0, 1, 0, 300.0, 2, "2021-07-03"),
        ])
        # Generate random binary target values
        y = np.random.randint(2, size=3)

        model_params = {
            'n_estimators': 10,
            'max_depth': 5,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42
        }

        pipeline = ModelTrainPipeline.create_train_pipeline(model_params=model_params)
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        # Assert that the predictions are binary values (0 or 1)
        self.assertTrue(np.all(np.isin(y_pred, [0, 1])))


if __name__ == '__main__':
    unittest.main()
