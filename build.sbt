name := "hello"

version := "1.0"

scalaVersion := "2.11.8"

val sparkVersion = "1.6.1"

resolvers ++= Seq(
  "MMLSpark Repo" at "https://mmlspark.azureedge.net/maven"
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.databricks" % "spark-csv_2.10" % "1.5.0",
  "com.microsoft.ml.spark" %% "mmlspark" % "1.0.0-rc1"
)