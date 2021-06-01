import com.microsoft.ml.spark.lightgbm.LightGBMClassifier
import org.apache.log4j._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.Pipeline

object MLlibClassifier {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("simple").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    // read dataset
    val df = sqlContext.read.format("com.databricks.spark.csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("sep", ",")
      .load("src/main/resources/creditcard.csv")

    // get only columns v1 to v28 and amount as features
    val columnName = Seq("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount")
    val featureCols = df.select(columnName.map(name => col(name)): _*)

    // set input and output cols
    val assembler = new VectorAssembler()
      .setInputCols(featureCols.columns)
      .setOutputCol("features")

    // hyperparameters determined in https://www.kaggle.com/patelatharva/credit-card-transaction-fraud-detection
    val lgb = new LightGBMClassifier()
      .setLearningRate(0.1)
      .setEarlyStoppingRound(100)
      .setFeaturesCol("features")
      .setLabelCol("Class")
      .setIsUnbalance(true)
      .setBaggingFraction(0.8)
      .setBaggingFreq(1)
      .setFeatureFraction(0.944714847210862)
      .setLambdaL1(1.0)
      .setLambdaL2(45.0)
      .setMaxBin(60)
      .setMaxDepth(58)
      .setNumIterations(379)
      .setNumLeaves(850)
      .setObjective("binary")
      .setBaggingSeed(7)

    // create pipeline with assembler (input/output cols) and classifier
    val pipeline = new Pipeline().setStages(Array(assembler, lgb))

    // split dataset, 80% train and 20% test
    val Array(train, test) = df.randomSplit(Array(0.8, 0.2), seed = 7)

    // train the model with train data
    val model = pipeline.fit(train)

    // predict with test data
    val predictions = model.transform(test)

    // show first 10 predictions
    predictions.select("Class", "prediction", "probability").show(10)

    // calculate test area under ROC
    val binaryEvaluator = new BinaryClassificationEvaluator().setLabelCol("Class").setMetricName("areaUnderROC")
    val testUnderROC = binaryEvaluator.evaluate(predictions)
    println("Test Area Under ROC: %.5f\n".format(testUnderROC))

    // get the tp's, tn's, fp's, fn's
    val tp = predictions.filter(col("Class").equalTo(1) && col("prediction").equalTo(1.0)).count()
    val tn = predictions.filter(col("Class").equalTo(0) && col("prediction").equalTo(0.0)).count()
    val fp = predictions.filter(col("Class").equalTo(0) && col("prediction").equalTo(1.0)).count()
    val fn = predictions.filter(col("Class").equalTo(1) && col("prediction").equalTo(0.0)).count()
    println("TP: %s".format(tp))
    println("TN: %s".format(tn))
    println("FP: %s".format(fp))
    println("FN: %s".format(fn))
    println("Total: %s\n".format(predictions.count()))

    // calculate recall
    var recall: Double = -1.0
    if (tp + fn > 0) {
      recall = tp.toDouble / (tp.toDouble + fn.toDouble)
      println("Recall: %.5f".format(recall))
    }

    // calculate precision
    var precision: Double = -1.0
    if (tp + fp > 0) {
      precision = tp.toDouble / (tp.toDouble + fp.toDouble)
      println("Precision: %.5f".format(precision))
    }

    // calculate precision
    if (recall != -1.0 && precision != -1.0) {
      val f1 = (2 * precision * recall) / (precision + recall)
      println("F1: %.5f".format(f1))
    }
  }
}
