package com.ml.classification;

import org.apache.spark.ml.feature.StringIndexerModel;
import static org.apache.spark.sql.functions.col;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.MaxAbsScaler;
import org.apache.spark.ml.feature.MaxAbsScalerModel;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

public class TwitterUserGenderClassification {


	/*Setting hyper parameters for classification models. Setting MaxDepth as 12 for optimal accuracy
	 * Have checked for all numbers between - 5 and 25. maxDepth = 12 yields a good result
	 * maxDepth = 15 also gave fair results.
	 * with maxDepth 7 and below , the model is UNDERFITTED.
	 * Got very good accuracy in training data with maxDepth = 25 but it led to OVERFITTING model.
	 */
	public static int maxDepth = 12;
	public static int minInfoGain = 0;
	public static int minInstancesPerNode = 4;
	private static StringIndexerModel indexerModel_gender = null;

	// Change the input and output paths to your requirement 
	// Default input path of gender classification data
	private static String inputfilePath = "data/gender-classifier-DFE-791531.csv";
	// Default output path for output results
	private static String outputFilePath = "output";


	//main method starts now
	public static void main(String[] args) {

		// setting the error logs
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

		// setting up the spark session
		SparkSession sparkSession = SparkSession.builder()  //SparkSession  
				.appName("TwitterGenderClassification") 
				.master("local[*]") 
				.getOrCreate();

		//Reading Data from a CSV file 
		//Inferring Schema and Setting Header as True
		//Picking only those columns which are required for the processing
		Dataset<Row> rawData = sparkSession.read().option("header", true).option("inferSchema",true).csv(inputfilePath).
				select(col("_unit_id"),col("gender"),col("gender:confidence"),col("description"),col("gender_gold"),
						col("link_color"),col("name"),col("sidebar_color"),col("text")).where(col("gender").isNotNull());

		System.out.println("Total records : " + rawData.count());

		TwitterUserGenderClassification twitterGenderClassification = new TwitterUserGenderClassification();

		// Calls gender identification method.
		System.out.println("TwitterUserGenderClassification Program is starting now...");
		twitterGenderClassification.genderIdentification(rawData);
		System.out.println("TwitterUserGenderClassification ends.....");
	}

	private void genderIdentification(Dataset<Row> rawData) {

		//Initializing the data sets to null 
		Dataset<Row> dsTrainingData = null;
		Dataset<Row> dsTestData = null;
		Dataset<Row> rfTrainingData = null;
		Dataset<Row> rfTestData = null;

		//Calling conversion method on input data set
		Dataset<Row> convertedDS = conversion(rawData);

		Dataset<Row> df= convertedDS.select(col("gender_IDX").as("label"),
				col("_unit_id"),col("gender"), col("gender_IDX"),col("link_color_IDX"),col("sidebar_color_IDX"),col("description_feature"),
				col("informalcontent_feature"));

		//Vector assembler is assembling all the required columns that will be used as features.
		//The columns chosen as features are already converted to desired numeric form
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{"link_color_IDX","sidebar_color_IDX","description_feature","informalcontent_feature"})
				.setOutputCol("features");


		Dataset<Row> LRdf = assembler.transform(df).select("_unit_id","gender","gender_IDX","label","features");

		//Tried to implement the features directly in Decision Tree Model
		//However, it gave errors and then had to go for scaling and normalizing

		System.out.println("Scaling the features now....");

		// scale the features using MaxAbsScaler
		MaxAbsScaler scaler = new MaxAbsScaler().setInputCol("features").setOutputCol("scaledFeatures");
		MaxAbsScalerModel scalerModel = scaler.fit(LRdf);
		LRdf = scalerModel.transform(LRdf);

		System.out.println("Normalizing the features now....");
		// normalize the scaled features received in previous step
		Normalizer normalizer = new Normalizer().setInputCol("scaledFeatures").setOutputCol("normalizedFeatures").setP(1.0);
		LRdf = normalizer.transform(LRdf).select("_unit_id","gender","gender_IDX","label","normalizedFeatures");

		// Preparing the data sets for Decision Tree Classification Model Now.... 
		System.out.println("Preparing the data sets for Decision Tree Classification Model Now....");
		Dataset<Row>[] dsDecisionTree=featureVectorization(LRdf);

		dsTrainingData=dsDecisionTree[0];
		dsTestData=dsDecisionTree[1];
		dsTrainingData.persist();
		dsTestData.persist();

		//Passing the training and test data set to Decision Tree Classification algorithm now
		System.out.println("Decision Tree Classification Algorithm working on Data Sets now.....");
		decisionTreeClassificationModel(dsTrainingData , dsTestData);

		//// Preparing the data sets for Random Forest Classification Model Now....
		System.out.println("Preparing the data sets for Random Forest Classification Model Now....");
		Dataset<Row>[] dsRandomForest=featureVectorization(LRdf);
		rfTrainingData=dsRandomForest[0];
		rfTestData=dsRandomForest[1];
		rfTrainingData.persist();
		rfTestData.persist();

		//Passing the training and test data set to Random Forest Classification algorithm now
		System.out.println("Random Forest Classification Algorithm working on Data Sets now.....");
		randomForestClassificationModel(rfTrainingData , rfTestData);
	}

	private Dataset<Row> conversion(Dataset<Row> rawData) {

		System.out.println("The columns are being converted into required datatypes / format now.....");

		//Converting the gender:confidence column to double
		rawData = rawData.withColumn("gender_confidence", rawData.col("gender:confidence").cast(DataTypes.DoubleType));

		//interested in records where gender confidence is greater or equal to 1
		//Or gender gold column in not null
		rawData = rawData.filter(rawData.col("gender:confidence").geq(1).or(rawData.col("gender_gold").isNotNull()))
				.drop("gender:confidence", "gender_gold");

		//Looking at those records where gender is either male , female , brand. Ignoring 'unknown' as 
		//it will not help in training the model or in predicting
		rawData = rawData.where("gender in('male','female','brand')");

		System.out.println("Converting column values to numeric index using StringIndexer...");

		//Convert a String column 'link_color' to numeric using StringIndexer
		StringIndexer indexer1 = new StringIndexer().setInputCol("link_color").setOutputCol("link_color_IDX");
		Dataset<Row> indexed_first = indexer1.setHandleInvalid("skip").fit(rawData).transform(rawData).drop("link_color");

		//Convert a String column 'sidebar_color' to numeric using StringIndexer
		StringIndexer indexer2 = new StringIndexer().setInputCol("sidebar_color").setOutputCol("sidebar_color_IDX");
		Dataset<Row> indexed_second = indexer2.setHandleInvalid("skip").fit(indexed_first).transform(indexed_first).drop("sidebar_color");

		//Convert String column 'gender' to numeric using StringIndexer
		StringIndexer indexer_gender = new StringIndexer().setInputCol("gender").setOutputCol("gender_IDX");
		indexerModel_gender = indexer_gender.setHandleInvalid("skip").fit(indexed_second);
		Dataset<Row> indexed_third = indexerModel_gender.transform(indexed_second);

		// This is to fill null values in String columns with blank like - "" 
		Dataset<Row> nullRemovedData = indexed_third.na().fill("");

		System.out.println("Tokeninizing , Removing Stopwords and calcuating Hashing TF & IDF.");


		// Tokenize column 'description'
		Tokenizer tokenizerDesc = new Tokenizer();
		tokenizerDesc.setInputCol("description").setOutputCol("desc_temp");
		Dataset<Row> tokenizedDesc = tokenizerDesc.transform(nullRemovedData).drop("description");

		// create StopWordsRemover object
		StopWordsRemover stopWordsRemoverDesc = new StopWordsRemover();
		stopWordsRemoverDesc.setInputCol("desc_temp").setOutputCol("desc_words");
		Dataset<Row> stopWordRemovedDesc = stopWordsRemoverDesc.transform(tokenizedDesc).drop("desc_temp");

		// Create the Term Frequency Matrix
		HashingTF hashingTFDesc = new HashingTF().setNumFeatures(1000);
		hashingTFDesc.setInputCol("desc_words").setOutputCol("desc_TF");
		Dataset<Row> hashingDesc = hashingTFDesc.transform(stopWordRemovedDesc).drop("desc_words");

		// Calculate the Inverse Document Frequency
		IDF idfDesc = new IDF().setInputCol("desc_TF").setOutputCol("description_feature");
		IDFModel modelDesc = idfDesc.fit(hashingDesc);
		Dataset<Row> idfModelDesc = modelDesc.transform(hashingDesc).drop("desc_TF");

		//Final column achieved here is - "description_feature"

		// Tokenize column 'text'
		Tokenizer tokenizerText = new Tokenizer();
		tokenizerText.setInputCol("text").setOutputCol("text_temp");
		Dataset<Row> tokenizedText = tokenizerText.transform(idfModelDesc).drop("text"); 


		// create StopWordsRemover object
		StopWordsRemover stopWordsRemoverText = new StopWordsRemover();
		stopWordsRemoverText.setInputCol("text_temp").setOutputCol("text_words");
		Dataset<Row> stopWordRemovedText = stopWordsRemoverText.transform(tokenizedText).drop("text_temp");

		// Create the Term Frequency Matrix
		HashingTF hashingTFText = new HashingTF().setNumFeatures(1000);
		hashingTFText.setInputCol("text_words").setOutputCol("text_TF");
		Dataset<Row> hashingText = hashingTFText.transform(stopWordRemovedText).drop("text_words");

		// Calculate the Inverse Document Frequency
		IDF idfText = new IDF().setInputCol("text_TF").setOutputCol("informalcontent_feature");
		IDFModel modelText = idfText.fit(hashingText);
		Dataset<Row> idfModelText = modelText.transform(hashingText).drop("text_TF");

		//Final column achieved here is - "informalcontent_feature"

		return idfModelText;

	}

	private void randomForestClassificationModel(Dataset<Row> rfTrainingData, Dataset<Row> rfTestingData) {

		RandomForestClassifier randomForestClassifier = new RandomForestClassifier().setLabelCol("label")
				.setFeaturesCol("normalizedFeatures").setMaxDepth(maxDepth).setMinInfoGain(minInfoGain)
				.setMinInstancesPerNode(minInstancesPerNode).setSeed(0);
		RandomForestClassificationModel randomForestClassificationModel = randomForestClassifier.fit(rfTrainingData);

		Dataset<Row> rfPredictions = null;
		// Predict on training data
		rfPredictions = randomForestClassificationModel.transform(rfTrainingData);
		// evaluate the model
		System.out.println("Starting Random forest classification model evaluation using training data...");
		rfPredictions = modelEvaluation(rfPredictions);
		// Predict on testing data
		rfPredictions = randomForestClassificationModel.transform(rfTestingData);
		// evaluate the model
		System.out.println("Starting Random forest classification model evaluation using test data...");
		rfPredictions = modelEvaluation(rfPredictions);

		// print the evaluation result
		String outputDir = outputFilePath + "/random_forest_classification_output";
		rfPredictions.coalesce(1).write().option("header", true).mode(SaveMode.Overwrite).format("csv").save(outputDir);
	}

	private void decisionTreeClassificationModel(Dataset<Row> dsTrainingData, Dataset<Row> dsTestingData) {

		DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier().setLabelCol("label")
				.setFeaturesCol("normalizedFeatures").setMaxDepth(maxDepth).setMinInfoGain(minInfoGain)
				.setMinInstancesPerNode(minInstancesPerNode).setSeed(0);
		DecisionTreeClassificationModel decisionTreeClassificationModel = decisionTreeClassifier.fit(dsTrainingData);

		Dataset<Row> dsPredictions = null;

		// Predict on training data
		dsPredictions = decisionTreeClassificationModel.transform(dsTrainingData);
		System.out.println("Starting Decision tree classification model evaluation using training data...");
		// evaluate the model
		dsPredictions = modelEvaluation(dsPredictions);
		// Predict on testing data
		dsPredictions = decisionTreeClassificationModel.transform(dsTestingData);
		// evaluate the model
		System.out.println("Starting Decision tree classification model evaluation using test data...");
		dsPredictions = modelEvaluation(dsPredictions);

		// print the evaluation result
		String outputDir = outputFilePath + "/decision_tree_classification_output";
		dsPredictions.coalesce(1).write().option("header", true).mode(SaveMode.Overwrite).format("csv").save(outputDir);
	}

	private Dataset<Row> modelEvaluation(Dataset<Row> dsPredictions) {

		// Select (prediction, gender label) and compute.
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("gender_IDX")
				.setPredictionCol("prediction");


		// accuracy computation
		evaluator.setMetricName("accuracy");
		double accuracy = evaluator.evaluate(dsPredictions);
		System.out.println("Accuracy = " + Math.round(accuracy * 100) + " %");

		// weightedPrecision computation
		evaluator.setMetricName("weightedPrecision");
		double precision = evaluator.evaluate(dsPredictions);
		System.out.println("Precision = " + Math.round(precision * 100) + " %");

		// weightedRecall computation
		evaluator.setMetricName("weightedRecall");
		double recall = evaluator.evaluate(dsPredictions);
		System.out.println("Recall = " + Math.round(recall * 100) + " %");

		// F1 score computation
		evaluator.setMetricName("f1");
		double f1Score = evaluator.evaluate(dsPredictions);
		System.out.println("f1 score = " + Math.round(f1Score * 100) + " %");

		// transform indexed prediction to string format and prepare confusion matrix
		IndexToString converter = new IndexToString().setInputCol("prediction").setOutputCol("predicted_gender")
				.setLabels(indexerModel_gender.labels());


		Dataset<Row> confusionMatrix = converter.transform(dsPredictions);
		confusionMatrix = confusionMatrix.groupBy("gender", "predicted_gender").count().orderBy("gender","predicted_gender");

		/*
		 * I have specifically avoided showing confusion matrix here. If you want to see 
		 * please uncomment below lines :
		System.out.println("Confusion Matrix is as follows :  ....");
		confusionMatrix.show();
		 */

		return confusionMatrix;
	}


	private Dataset<Row>[] featureVectorization(Dataset<Row> record) {

		// split the data randomly in two parts (training and testing)
		Dataset<Row>[] datasetAll = record.randomSplit(new double[] { 0.7, 0.3 });
		// fetch the training data in a dataset
		Dataset<Row> trainingData = datasetAll[0];
		// fetch the testing data in a dataset
		Dataset<Row> testingData = datasetAll[1];

		Dataset<Row>[] featuredDataSets= new Dataset[2];
		featuredDataSets[0]=trainingData;
		featuredDataSets[1]=testingData;
		return featuredDataSets;
	}
}

