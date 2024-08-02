using Microsoft.ML;

class Program
{
    static void Main(string[] args)
    {
        var context = new MLContext();

        // Simulated data for demonstration purposes
        var data = new List<SensorData>
        {
            new SensorData { Time = DateTime.Parse("2024-03-08 00:00"), Value = 1.0f },
            new SensorData { Time = DateTime.Parse("2024-03-08 01:00"), Value = 1.1f },
            new SensorData { Time = DateTime.Parse("2024-03-08 02:00"), Value = 1.0f },
            new SensorData { Time = DateTime.Parse("2024-03-08 03:00"), Value = 1.2f },
            new SensorData { Time = DateTime.Parse("2024-03-08 04:00"), Value = 1.1f },
            new SensorData { Time = DateTime.Parse("2024-03-08 05:00"), Value = 5.0f }, // Anomaly
            new SensorData { Time = DateTime.Parse("2024-03-08 06:00"), Value = 1.1f },
            new SensorData { Time = DateTime.Parse("2024-03-08 07:00"), Value = 1.0f },
            new SensorData { Time = DateTime.Parse("2024-03-08 08:00"), Value = 1.2f },
            new SensorData { Time = DateTime.Parse("2024-03-08 09:00"), Value = 1.0f }
        };

        var dataView = context.Data.LoadFromEnumerable(data);

        var pipeline = context.Transforms.DetectSpikeBySsa(
            outputColumnName: nameof(AnomalyPrediction.Prediction),
            inputColumnName: nameof(SensorData.Value),
            confidence: 95.0,
            pvalueHistoryLength: data.Count / 2,
            trainingWindowSize: data.Count,
            seasonalityWindowSize: data.Count / 4);

        var model = pipeline.Fit(dataView);
        var transformedData = model.Transform(dataView);

        var predictions = context.Data.CreateEnumerable<AnomalyPrediction>(transformedData, reuseRowObject: false).ToList();

        Console.WriteLine("Time\t\tValue\tAnomaly");
        for (int i = 0; i < predictions.Count; i++)
        {
            var sensorData = data[i];
            var prediction = predictions[i];
            var isAnomaly = prediction.Prediction[0] == 1;
            Console.WriteLine($"{sensorData.Time}\t{sensorData.Value}\t{isAnomaly}");
        }
    }
}