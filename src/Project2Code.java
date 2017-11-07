import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.*;
import java.io.*;
import java.util.function.DoubleBinaryOperator;

import com.sun.tools.doclets.formats.html.SourceToHTMLConverter;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.converters.ConverterUtils;

public class Project2Code {
    static double vectors[][][] = new double[60][][];//50 documents, unfixed sentences, and 300 is vector size
    static ArrayList<Node> arrayOfDocuments[] = new ArrayList [60];
    static double edges[][][] = new double[60][][];
    public static class Sentence{
        int ID;
        String docName;
        String tag;
        String sent;
        String root;
        int advmod_pres;
        int acomp_pres;
        int xcomp_pres;
         public Sentence(int ID, String docName, String sent, String root, int advmod_pres, int acomp_pres, int xcomp_pres, String tag){
            this.ID = ID;
            this.docName = docName;
            this.sent = sent;
            this.advmod_pres = advmod_pres;
            this.acomp_pres = acomp_pres;
            this.xcomp_pres = xcomp_pres;
            this.root = root;
            this.tag = tag;
         }

    }

    public static class ClassifySentence{
        int ID;
        double probabilityO;
        double probabilityF;
        String classify;
        String doc;
        ClassifySentence(int ID, double probabilityO, double probabilityF, String classify, String doc){
            this.ID = ID;
            this.probabilityO = probabilityO;
            this.probabilityF = probabilityF;
            this.classify = classify;
            this.doc = doc;
        }
    }

    public static class SentenceStats{
        int positivePolarCount;
        int negativePolarCount;
        int polarityOfRoot;
        int advMod;
        int aComp;
        int xComp;
        String docName;
        int ID;
        String tag;
        public SentenceStats(int ID, String docName, int positivePolarCount, int negativePolarCount, int polarityOfRoot, int advMod, int aComp, int xComp, String tag){
            this.positivePolarCount = positivePolarCount;
            this.negativePolarCount = negativePolarCount;
            this.polarityOfRoot = polarityOfRoot;
            this.advMod = advMod;
            this.aComp = aComp;
            this.xComp = xComp;
            this.docName = docName;
            this.ID = ID;
            this.tag = tag;
        }
    }

    public static class Node {//create a node for each sentence.
        double hub;
        double authority;

        public Node(double hub, double authority){
            this.hub = hub;
            this.authority = authority;
        }
    }

    public static void main(String[] args) throws Exception{




        File dirVectors = new File("/Users/parshwashah/Desktop/project2/train_data/train_vectors");



        File dirTest = new File("/Users/parshwashah/Desktop/project2/test_data/test_files");
        File [] filesTest = dirTest.listFiles();
        File dirTraining = new File("/Users/parshwashah/Desktop/project2/train_data/train_files");
        File [] filesTraining = dirTraining.listFiles();

        ArrayList<Sentence> sentenceData = new ArrayList<>();//Arraylist of all the sentences which will be harvested from files, and data will be analysed.
        ArrayList<String> positivePolar = extractWords(new File("/Users/parshwashah/Desktop/project2/pos_polar.txt"));//Arraylist of all the positive polar words
        ArrayList<String> negativePolar = extractWords(new File("/Users/parshwashah/Desktop/project2/neg_polar.txt"));//Arraylist of all the negative polar words
        ArrayList<SentenceStats> sentenceStats = new ArrayList<>();

        for(File f:filesTest){
            //System.out.println(f.getName());
            String fileName = f.getName();//get the File name
            FileInputStream fis = new FileInputStream(f);
            BufferedReader br = new BufferedReader(new InputStreamReader(fis));
            String line = null;
            int counter = 0;
            int k =0;
            br.readLine();
            while ((line = br.readLine()) != null) {//extract lines of the file
                //if(k==0){
                   // k++;
                    //continue;
               // }else{//do something with data
                    //System.out.println(line);
                    String arr[] = line.split("\t"); //get all the attributes
                    int ID = Integer.parseInt(arr[0]);
                    String docName = f.getName();
                    String sent = arr[1];
                    String root = arr[2];
                    int advmod_pres = Integer.parseInt(arr[3]);
                    int acomp_pres = Integer.parseInt(arr[4]);
                    int xcomp_pres = Integer.parseInt(arr[5]);
                    Sentence sentence = new Sentence(ID, docName, sent.toLowerCase(), root, advmod_pres, acomp_pres, xcomp_pres, "NA");
                    sentenceData.add(sentence);

                //}
            }

            br.close();
        }

        int fie = 0;

        for(File f:filesTraining){
            //System.out.println(f.getName());
            String fileName = f.getName();//get the File name
            //System.out.println(fileName);
            FileInputStream fis = new FileInputStream(f);
            BufferedReader br = new BufferedReader(new InputStreamReader(fis));
            String line = null;

            int k = 0;
            br.readLine();
            while ((line = br.readLine()) != null) {//extract lines of the file

                //do something with data
                    //System.out.println(line);
                    String arr[] = line.split("\t"); //get all the attributes
                    int ID = Integer.parseInt(arr[0]);
                    String docName = f.getName();
                    String tag = arr[1];
                    String sent = arr[2];
                    String root = arr[3];
                    int advmod_pres = Integer.parseInt(arr[4]);
                    int acomp_pres = Integer.parseInt(arr[5]);
                    int xcomp_pres = Integer.parseInt(arr[6]);
                    Sentence sentence = new Sentence(ID, docName, sent.toLowerCase(), root.toLowerCase(), advmod_pres, acomp_pres, xcomp_pres, tag);
                    sentenceData.add(sentence);
                    k++;
            }
            //vectors[fie] = new double[k][300];
            fie++;

            br.close();
        }
       // System.out.println(fie);
        for(Sentence sentence: sentenceData){
           // System.out.println(sentence.docName);
            int positivePolarCount = 0;
            int negativePolarCount = 0;
            String sent = sentence.sent;
            String words[] = sent.split(" ");
            //System.out.println(sent);
            for(String w:words){
               //System.out.println(w);
                if(positivePolar.contains(w)){
                    positivePolarCount++;
                }
                if(negativePolar.contains(w)){
                    negativePolarCount++;
                }
            }
            String name = sentence.docName;
            int ID = sentence.ID;
            String root = sentence.root;
            int polarityOfRoot = 0;
            if(positivePolar.contains(root)){
                polarityOfRoot = 1;
            }else if(negativePolar.contains(root)){
                polarityOfRoot = -1;
            }
            int advMod = sentence.advmod_pres;
            int aComp = sentence.acomp_pres;
            int xComp = sentence.xcomp_pres;
            SentenceStats stats = new SentenceStats(ID, name, positivePolarCount, negativePolarCount,polarityOfRoot,advMod,aComp,xComp,sentence.tag);
            sentenceStats.add(stats);

        }
        //System.out.println(sentenceStats.size()); 3099 OK.

        HashMap<String, Integer> mapPosTest = new HashMap<>();
        HashMap<String, Integer> mapNegTest = new HashMap<>();
        HashMap<String, Integer> mapPosTrain = new HashMap<>();
        HashMap<String, Integer> mapNegTrain = new HashMap<>();

        for(SentenceStats stats: sentenceStats){
            if(stats.docName.contains("test")) {
                if (mapPosTest.containsKey(stats.docName) == false) {
                    mapPosTest.put(stats.docName, stats.positivePolarCount);
                } else if (mapPosTest.containsKey(stats.docName) == true) {
                    int val = mapPosTest.get(stats.docName);
                    val = val + stats.positivePolarCount;
                    mapPosTest.put(stats.docName, val);
                }

                if (mapNegTest.containsKey(stats.docName) == false) {
                    mapNegTest.put(stats.docName, stats.negativePolarCount);
                } else if (mapNegTest.containsKey(stats.docName) == true) {
                    int val = mapNegTest.get(stats.docName);
                    val = val + stats.negativePolarCount;
                    mapNegTest.put(stats.docName, val);
                }
            }else{
                if (mapPosTrain.containsKey(stats.docName) == false) {
                    mapPosTrain.put(stats.docName, stats.positivePolarCount);
                } else if (mapPosTrain.containsKey(stats.docName) == true) {
                    int val = mapPosTrain.get(stats.docName);
                    val = val + stats.positivePolarCount;
                    mapPosTrain.put(stats.docName, val);
                }

                if (mapNegTrain.containsKey(stats.docName) == false) {
                    mapNegTrain.put(stats.docName, stats.negativePolarCount);
                } else if (mapNegTrain.containsKey(stats.docName) == true) {
                    int val = mapNegTrain.get(stats.docName);
                    val = val + stats.negativePolarCount;
                    mapNegTrain.put(stats.docName, val);
                }
            }
        }

        Set<String> setPosTest = mapPosTest.keySet();
        Set<String> setNegTest = mapNegTest.keySet();
        Set<String> setPosTrain = mapPosTrain.keySet();
        Set<String> setNegTrain = mapNegTrain.keySet();

        findMax(setPosTest,mapPosTest);
        findMax(setNegTest,mapNegTest);
        findMax(setPosTrain,mapPosTrain);
        findMax(setNegTrain,mapNegTrain);

        createArffFile("dataArff.arff",sentenceStats);

    }

    public static void createArffFile(String name, ArrayList<SentenceStats> sentenceStats)throws Exception{
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(name));
        bufferedWriter.write("@relation trainingdata\n\n");
        bufferedWriter.write("@attribute Opinion/Factual {O,F}\n");
        bufferedWriter.write("@attribute PositivePolarWordCount numeric\n");
        bufferedWriter.write("@attribute NegativePolarWordCount numeric\n");
        bufferedWriter.write("@attribute RootPolarity {-1,0,1}\n");
        bufferedWriter.write("@attribute advModDependency {0,1}\n");
        bufferedWriter.write("@attribute aCompDependency {0,1}\n");
        bufferedWriter.write("@attribute xCompDependency {0,1}\n");
        bufferedWriter.write("\n@data\n");
        //BufferedWriter bufferedWriter1 = null;
        ArrayList<String> testFiles = new ArrayList<>();//keeps track of what files are created.
        HashMap<String, BufferedWriter> map = new HashMap<>();
        map.put("test_0.tsv",new BufferedWriter(new FileWriter("test_0.arff")));
        map.put("test_1.tsv",new BufferedWriter(new FileWriter("test_1.arff")));
        map.put("test_2.tsv",new BufferedWriter(new FileWriter("test_2.arff")));
        map.put("test_3.tsv",new BufferedWriter(new FileWriter("test_3.arff")));
        map.put("test_4.tsv",new BufferedWriter(new FileWriter("test_4.arff")));
        map.put("test_5.tsv",new BufferedWriter(new FileWriter("test_5.arff")));
        map.put("test_6.tsv",new BufferedWriter(new FileWriter("test_6.arff")));
        map.put("test_7.tsv",new BufferedWriter(new FileWriter("test_7.arff")));
        map.put("test_8.tsv",new BufferedWriter(new FileWriter("test_8.arff")));
        map.put("test_9.tsv",new BufferedWriter(new FileWriter("test_9.arff")));
        for(SentenceStats s: sentenceStats){
            if(s.tag.equals("NA")==false){//training data
                bufferedWriter.write(s.tag+","+s.positivePolarCount+","+s.negativePolarCount+","+s.polarityOfRoot+","+s.advMod+","+s.aComp+","+s.xComp+"\n");
            }else{//test
                String n= s.docName;
                BufferedWriter wr = map.get(n);
                if(testFiles.contains(n)==false){
                    testFiles.add(n);
                    wr.write("@relation testingdata\n\n");
                    wr.write("@attribute Opinion/Factual {O,F}\n");
                    wr.write("@attribute PositivePolarWordCount numeric\n");
                    wr.write("@attribute NegativePolarWordCount numeric\n");
                    wr.write("@attribute RootPolarity {-1,0,1}\n");
                    wr.write("@attribute advModDependency {0,1}\n");
                    wr.write("@attribute aCompDependency {0,1}\n");
                    wr.write("@attribute xCompDependency {0,1}\n");
                    wr.write("\n@data\n");
                    wr.write("?,"+s.positivePolarCount+","+s.negativePolarCount+","+s.polarityOfRoot+","+s.advMod+","+s.aComp+","+s.xComp+"\n");
                }else{
                    wr.write("?,"+s.positivePolarCount+","+s.negativePolarCount+","+s.polarityOfRoot+","+s.advMod+","+s.aComp+","+s.xComp+"\n");
                }
            }
        }

        ArrayList<String> trainFiles = new ArrayList<>();
        HashMap<String, BufferedWriter> map1 = new HashMap<>();

        for(int i=0;i<50;i++){
            String k = "train_"+i+".tsv";
            String ar = "train_"+i+".arff";
            map1.put(k, new BufferedWriter(new FileWriter(ar)));//make the map
        }

        for(SentenceStats s:sentenceStats){
            if(s.tag.equals("NA")==false){//create seperate ARFF files for train data
                String n = s.docName;
                BufferedWriter wr = map1.get(n);
                if(trainFiles.contains(n)==false){
                    trainFiles.add(n);
                    wr.write("@relation trainingdata\n\n");
                    wr.write("@attribute Opinion/Factual {O,F}\n");
                    wr.write("@attribute PositivePolarWordCount numeric\n");
                    wr.write("@attribute NegativePolarWordCount numeric\n");
                    wr.write("@attribute RootPolarity {-1,0,1}\n");
                    wr.write("@attribute advModDependency {0,1}\n");
                    wr.write("@attribute aCompDependency {0,1}\n");
                    wr.write("@attribute xCompDependency {0,1}\n");
                    wr.write("\n@data\n");
                    wr.write("?,"+s.positivePolarCount+","+s.negativePolarCount+","+s.polarityOfRoot+","+s.advMod+","+s.aComp+","+s.xComp+"\n");
                }else{
                    wr.write("?,"+s.positivePolarCount+","+s.negativePolarCount+","+s.polarityOfRoot+","+s.advMod+","+s.aComp+","+s.xComp+"\n");
                }
            }
        }


        bufferedWriter.close();
        ArrayList<ClassifySentence> allsentences = new ArrayList<>();
        //double edges[][][] = new double[60][][];
        Set<String> keys = map.keySet();
        Set<String> keys1 = map1.keySet();
        for(String s:keys){
            map.get(s).close();
        }
        for(String s:keys1){
            map1.get(s).close();
        }

        ConverterUtils.DataSource source1 = new ConverterUtils.DataSource("/Users/parshwashah/Desktop/project2/dataArff.arff");
        Instances train = source1.getDataSet();
        if (train.classIndex() == -1)
            train.setClassIndex(train.numAttributes() - 1);
        train.setClassIndex(0);

        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(train);

        //ArrayList<Node> arrayOfDocuments[] = new ArrayList [60];//store nodes of each document.
        String arr[] = {"test_0.arff","test_1.arff","test_2.arff","test_3.arff","test_4.arff","test_5.arff"
                        ,"test_6.arff","test_7.arff","test_8.arff","test_9.arff"};

        String arr1[] = new String[50];
        String arr2[] = new String[50];
        String arr3[] = new String[10];
       //String vectorFiles[] = new String[50];
        for(int i=0;i<arr1.length;i++){
            String x = "train_"+i+".arff";
            arr1[i] = x;
            String x2 = "/Users/parshwashah/Desktop/project2/train_data/train_vectors/train_vec_"+i+".tsv";//ALL THE VECTOR FILES.
            arr2[i] = x2;
        }

        for(int i=0;i<10;i++){
            arr3[i] = "/Users/parshwashah/Desktop/project2/test_data/test_vectors/test_vec_"+i+".tsv";
        }

        for(int i = 0; i < 60;i ++){
            arrayOfDocuments[i] = new ArrayList<>();
        }


        int ctrTest = 50;
        for(String doc:arr) {
            ArrayList<Node> x = new ArrayList<>();
            ConverterUtils.DataSource source2 = new ConverterUtils.DataSource("/Users/parshwashah/Desktop/project2/"+doc);
            System.out.println(doc);
            Instances test = source2.getDataSet();
            int num = test.numInstances();
            vectors[ctrTest] = new double[num][300];
            edges[ctrTest] = new double[num][num];
            //System.out.println(num);
            double maxProb = 0.0;
            ClassifySentence maxSentence = null;
            for (int i = 0; i < num; i++) {
                if (test.classIndex() == -1) {
                    test.setClassIndex(test.numAttributes() - 1);
                }
                test.setClassIndex(0);
                double probability[] = naiveBayes.distributionForInstance(test.instance((i)));

                x.add(new Node(probability[0], probability[1]));

                double label = naiveBayes.classifyInstance(test.instance(i));
                test.instance(i).setClassValue(label);
                if(probability[0]>maxProb){
                    maxProb = probability[0];
                    maxSentence = new ClassifySentence(i+1,probability[0],probability[1],test.instance(i).stringValue(0),doc);
                }
                //System.out.println(test.instance(i).stringValue(0) + " " + probability[0] + " " + probability[1]);
            }
            System.out.println("SENTENCESfoo: " +x.size());
            arrayOfDocuments[ctrTest] = x;
            ctrTest++;
            allsentences.add(maxSentence);
        }



        int counter = 0;
        for(String doc:arr1){
            ArrayList<Node> x = new ArrayList<>();
            ConverterUtils.DataSource source2 = new ConverterUtils.DataSource("/Users/parshwashah/Desktop/project2/"+doc);
            System.out.println(doc);
            Instances test = source2.getDataSet();
            int num = test.numInstances();
            vectors[counter] = new double[num][300];//initialize the vectors array.
            edges[counter] = new double[num][num];

            for(int i=0;i<num;i++){
                if (test.classIndex() == -1) {
                    test.setClassIndex(test.numAttributes() - 1);
                }
                test.setClassIndex(0);
                double probability[] = naiveBayes.distributionForInstance(test.instance((i)));
                x.add(new Node(probability[0], probability[1]));
                double label = naiveBayes.classifyInstance(test.instance(i));
                test.instance(i).setClassValue(label);
                //System.out.println(test.instance(i).stringValue(0) + " " + probability[0] + " " + probability[1]);
            }
            System.out.println("SENTENCES: "+x.size());
            arrayOfDocuments[counter] = x;
            counter ++;
        }

        int ctrr = 0;
        for(String fileName:arr2){

            FileInputStream fis = new FileInputStream(fileName);
            BufferedReader br = new BufferedReader(new InputStreamReader(fis));
            String line = null;
            br.readLine();//skip first line.
            int c = 0;
            while((line = br.readLine())!=null){
                String array[] = line.split("\t");
                String vec = array[1];
                vec = vec.substring(1,vec.length()-1);
                //System.out.println(vec);
                double[] numbers = Arrays.stream(vec.split(",")).mapToDouble(Double::parseDouble).toArray();
                //System.out.println(numbers.length);
                //vectors[ctr][counter] = numbers;
                vectors[ctrr][c] = numbers;
                c++;
            }
            //System.out.println(vectors[ctrr].length);
            ctrr++;

        }

        for(String fileName:arr3){
            System.out.println(fileName);
            FileInputStream fis = new FileInputStream(fileName);
            BufferedReader br = new BufferedReader(new InputStreamReader(fis));
            String line = null;
            br.readLine();//skip first line.
            int c = 0;
            while((line = br.readLine())!=null){
                String array[] = line.split("\t");
                String vec = array[1];
                vec = vec.substring(1,vec.length()-1);
                //System.out.println(vec);
                double[] numbers = Arrays.stream(vec.split(",")).mapToDouble(Double::parseDouble).toArray();
                //System.out.println(numbers.length);
                //vectors[ctr][counter] = numbers;
                vectors[ctrr][c] = numbers;
                c++;
            }
            System.out.println(vectors[ctrr].length);
            ctrr++;

        }

        for(int i=0;i<60;i++){//CREATE GRAPHS, for training data.
            ArrayList<Node> arrayList = arrayOfDocuments[i];//get sentences and their nodes from the doc.

            for(int j=0;j<arrayList.size();j++){
                for(int k = 0; k < arrayList.size();k++){
                    if(j==k){
                        edges[i][j][k]=0;
                    }else{
                        double similarity = cosSimilarity(vectors[i][j],vectors[i][k]);
                        double hit = arrayList.get(j).hub;
                        int d = j - k;
                        if(d<0){
                            d = -1 * d;
                        }
                        double wt = similarity * similarity * Math.pow(hit,3) * (1 + (1.0/d));

                        //wt = (double)Math.round(wt*100)/100;
                        edges[i][j][k] = wt;
                    }
                }
            }
        }

       // ArrayList<Node> n = arrayOfDocuments[50];
        /*
        for(int i=59;i<=59;i++){
            System.out.println("\nFILE: "+i);
            for(int j = 0; j<edges[i].length;j++){
                for(int k=0;k<edges[i].length;k++){
                   System.out.printf("%.2f ",edges[i][j][k]);
                }
                System.out.println();
            }
            System.out.println();
        }*/

        hits1(10000);
        //hits1(5000);
        //hits1(1000);
        int x =0;
        System.out.println("Maximum Opinionated Sentences in each document: ");
        for(ClassifySentence sent:allsentences){
            System.out.println("test_"+x+".tsv with sentence number: "+sent.ID+", with Probability of Opinionated "+sent.probabilityO+" ");
            x++;
        }
        int [] nodeIndex = new int[10];//store the max
        System.out.println("MAX opinionated sentences for HITS: ");
        for(int i = 50 ; i < 60 ; i ++){
            ArrayList<Node> sent = arrayOfDocuments[i];
            double max = 0 ;
            int maxSentenceID = 0;
            for(Node node:sent){
                if(node.hub > max){
                    max = node.hub;
                    maxSentenceID = sent.indexOf(node);
                    nodeIndex[i-50] = maxSentenceID;
                }

            }
           System.out.println("HITS DOC: test_"+(i-50)+", Sentence: "+(maxSentenceID+1));
        }

        for(int ind : nodeIndex)
            System.out.println(ind);

        for(int i = 50 ; i < 60 ; i ++){
            int index = nodeIndex[i-50];//get index for max sentence
            ArrayList<Node> sentences = arrayOfDocuments[i];
            ArrayList<Double> mS = new ArrayList<>();
            double max1 = 0;
            int ind1 = 0;
            double max2 = 0;
            int ind2 = 0;
            for(int j = 0 ; j < sentences.size(); j ++){
                if(index == j){
                    mS.add(Double.MIN_VALUE);
                    continue;
                }else {
                    double MScore = cosSimilarity(vectors[i][index], vectors[i][j]) * sentences.get(j).authority;
                    mS.add(MScore);
                }
            }

            for(double score:mS){
                if(score > max1){
                    ind1 = mS.indexOf(score);
                    max1 = score;
                }
            }

            mS.set(ind1, Double.MIN_VALUE);

            for(double score : mS){
                if(score > max2){
                    ind2 = mS.indexOf(score);
                    max2 = score;
                }
            }

            mS.set(ind2, Double.MIN_VALUE);
            System.out.println("test_"+(i-50)+"\nMax Opinionated Sentence: "+(index+1)+"\nTop Authority Sentence 1: "+(ind1+1)+" with M = "+max1+ "\nTop Authority Sentence 2: "+(ind2+1)+" with M = "+max2);

        }


    }

    static void hits1(int no) {

        for (int i = 50; i < 60; i++) {
            ArrayList<Node> sentences = arrayOfDocuments[i];
            double graph[][] = edges[i];//THIS IS ALL U NEED.

            for(int j = 1 ; j <= no ; j++){
                ArrayList<Node> newVals = new ArrayList<>();
                double sumHub = 0;
                double sumAuth = 0;
                for(int k=0;k<sentences.size();k++){
                    double sum1 = 0;
                    double sum2 = 0;
                    for(int m = 0; m < sentences.size(); m++){
                        if(m==k) continue;
                        sum1 = sum1 + (graph[k][m] * sentences.get(m).authority);
                        sum2 = sum2 + (graph[m][k] * sentences.get(m).hub);
                    }

                    newVals.add(new Node(sum1,sum2));
                    sumHub = sumHub + Math.pow(newVals.get(k).hub,2);
                    sumAuth = sumAuth + Math.pow(newVals.get(k).authority,2);
                }

                for(int k=0;k<sentences.size();k++){
                    newVals.get(k).hub = newVals.get(k).hub / Math.sqrt(sumHub);
                    newVals.get(k).authority = newVals.get(k).authority / Math.sqrt(sumAuth);
                    sentences.get(k).hub = newVals.get(k).hub;
                    sentences.get(k).authority = newVals.get(k).authority;
                }
            }
        }
    }


    static void findMax(Set<String> keys, HashMap<String, Integer> map){
        int max = 0;
        String doc = "";
        for(String s:keys){
            int t = map.get(s);
            if(t>max){
                max = t;
                doc = s;
            }
        }

        System.out.println(doc);
        System.out.println(max);
    }

    static double cosSimilarity(double[] arr1, double[] arr2){
        double s = 0;
        double s1 = 0;
        double s2 = 0;
        int gt01 = 0;
        int gt02 = 0;
        for(int i =0; i < arr1.length;i++){
            if(arr1[i]!=0){
                gt01++;
            }
            if(arr2[i]!=0){
                gt02++;
            }
            double f = arr1[i]*arr2[i];
            s = s + f;
            s1 = s1 + (arr1[i]*arr1[i]);
            s2 = s2 + (arr2[i]*arr2[i]);
        }

        if(gt01==0 || gt02==0)
            return 0;
        else
            return (s/Math.sqrt(s1 * s2));
    }

    static ArrayList<String> extractWords(File file) throws IOException{
        ArrayList<String> list = new ArrayList<>();
        FileInputStream fis = new FileInputStream(file);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        while ((line = br.readLine()) != null) {
            //System.out.println(line);
            list.add(line);
        }
        return list;
    }


}
