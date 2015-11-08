package ai.ia.agh.edu.pl.workshop.moandroid;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.TimingUtils;
import moa.streams.generators.RandomRBFGenerator;
import weka.core.Instance;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Experiment exp = new Experiment();
        exp.run(1000000, true);
    }

    public void run(int numInstances, boolean isTesting){
        Classifier learner = new HoeffdingTree();
        RandomRBFGenerator stream = new RandomRBFGenerator();
        stream.prepareForUse();

        learner.setModelContext(stream.getHeader());
        learner.prepareForUse();

        int numberSamplesCorrect = 0;
        int numberSamples = 0;
        boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        while (stream.hasMoreInstances() && numberSamples < numInstances) {
            Instance trainInst = stream.nextInstance();
            if (isTesting) {
                if (learner.correctlyClassifies(trainInst)){
                    numberSamplesCorrect++;
                }
            }
            numberSamples++;
            learner.trainOnInstance(trainInst);
        }
        double accuracy = 100.0 * (double) numberSamplesCorrect/ (double) numberSamples;
        double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()- evaluateStartTime);
        System.out.println(numberSamples + " instances processed with " + accuracy + "% accuracy in "+time+" seconds.");
    }
}
