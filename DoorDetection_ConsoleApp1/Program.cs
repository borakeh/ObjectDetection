using DoorDetection_ConsoleApp1;
using Microsoft.ML.Data;
using System.Diagnostics;
using System.Drawing;


/* Slow */
var image = MLImage.CreateFromFile(@"C:\Users\developer\Desktop\thedoor.png");
DoorDetection.ModelInput sampleData = new DoorDetection.ModelInput()
{
    Image = image,
};

Stopwatch stopwatch = Stopwatch.StartNew();

// Make a single prediction on the sample data and print results.
var predictionResult = DoorDetection.Predict(sampleData);

stopwatch.Stop();
Console.WriteLine(stopwatch.ElapsedMilliseconds);

Console.WriteLine("\n\nPredicted Boxes:\n");
if (predictionResult.PredictedBoundingBoxes == null)
{
    Console.WriteLine("No Predicted Bounding Boxes");
    return;
}


var boxes = predictionResult.PredictedBoundingBoxes.Chunk(4)
.Select(x => new { XTop = x[0], YTop = x[1], X2 = x[2], Y2 = x[3] }) // Replace XBottom and YBottom with X2 and Y2
.Zip(predictionResult.Score, (a, b) => new { Box = a, Score = b });

// Load the original image
Bitmap imageBitmap = new Bitmap(@"C:\Users\developer\Desktop\thedoor.png");

// Create a graphics object from the image
using (var graphics = Graphics.FromImage(imageBitmap))
{
    // Create a Pen object to style the bounding box
    using (var pen = new Pen(Color.CornflowerBlue, 2))
    {
        // Define a font and Brush for the text
        Font drawFont = new Font("Arial", 5);
        SolidBrush drawBrush = new SolidBrush(Color.Yellow);

        foreach (var item in boxes.Where(x => x.Score > .6))
        {
            int xTop = (int)item.Box.XTop;
            int yTop = (int)item.Box.YTop;
            int x2 = (int)item.Box.X2; // Replace xBottom with x2
            int y2 = (int)item.Box.Y2;  // Replace yBottom with y2

            int width = x2 - xTop;
            int height = y2 - yTop;

            // Ensure width and height are positive
            if (width < 0) width = 0;
            if (height < 0) height = 0;

            Console.WriteLine($"Drawing box at ({xTop}, {yTop}) with width {width} and height {height}");
            graphics.DrawRectangle(pen, xTop, yTop, width, height);

            // Draw score text above bounding box
            graphics.DrawString(item.Score.ToString(), drawFont, drawBrush, xTop, yTop - 15);
        }

        // Save the image with bounding boxes
        imageBitmap.Save(@"C:\Users\developer\Desktop\output.png");
        // Dispose of the font and brush objects after use
        drawFont.Dispose();
        drawBrush.Dispose();
    }
}