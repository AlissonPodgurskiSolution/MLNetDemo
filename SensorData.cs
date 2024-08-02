using Microsoft.ML.Data;

public class SensorData
{
    [LoadColumn(0)]
    public DateTime Time { get; set; }

    [LoadColumn(1)]
    public float Value { get; set; }
}

public class AnomalyPrediction
{
    [VectorType(3)]
    public double[] Prediction { get; set; }
}
