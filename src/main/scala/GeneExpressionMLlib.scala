import com.microsoft.ml.spark.lightgbm.LightGBMClassifier
import org.apache.log4j._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.col
import org.apache.spark.{SparkConf, SparkContext}

object GeneExpressionMLlib {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("simple").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)


    // read independent dataset
    val df = sqlContext.read.format("com.databricks.spark.csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("sep", ",")
      .load("src/main/resources/data_set.csv")

    val label = "cancer"
    val data = df.drop(col(label))

    // set input and output cols
    val assembler = new VectorAssembler()
      .setInputCols(data.columns)
      .setOutputCol("features")

    // set up a classifier
    val lgb = new LightGBMClassifier()
      .setLearningRate(0.1)
      .setEarlyStoppingRound(100)
      .setFeaturesCol("features")
      .setLabelCol(label)
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
    predictions.select("cancer", "prediction", "probability").show(10)

    // calculate test area under ROC
    val binaryEvaluator = new BinaryClassificationEvaluator().setLabelCol(label).setMetricName("areaUnderROC")
    val testUnderROC = binaryEvaluator.evaluate(predictions)
    println("Test Area Under ROC: %.5f\n".format(testUnderROC))

    // get the tp"s, tn"s, fp"s, fn"s
    val tp = predictions.filter(col(label).equalTo(1) && col("prediction").equalTo(1.0)).count()
    val tn = predictions.filter(col(label).equalTo(0) && col("prediction").equalTo(0.0)).count()
    val fp = predictions.filter(col(label).equalTo(0) && col("prediction").equalTo(1.0)).count()
    val fn = predictions.filter(col(label).equalTo(1) && col("prediction").equalTo(0.0)).count()
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
