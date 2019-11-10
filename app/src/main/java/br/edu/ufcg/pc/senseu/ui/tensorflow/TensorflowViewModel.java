package br.edu.ufcg.pc.senseu.ui.tensorflow;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

public class TensorflowViewModel extends ViewModel {

    private MutableLiveData<String> mText;

    public TensorflowViewModel() {
        mText = new MutableLiveData<>();
        mText.setValue("This is TensorFlow fragment");
    }

    public LiveData<String> getText() {
        return mText;
    }
}