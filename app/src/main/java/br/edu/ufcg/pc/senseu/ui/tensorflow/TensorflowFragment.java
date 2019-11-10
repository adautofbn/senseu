package br.edu.ufcg.pc.senseu.ui.tensorflow;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProviders;

import br.edu.ufcg.pc.senseu.R;

public class TensorflowFragment extends Fragment {

    private TensorflowViewModel dashboardViewModel;

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        dashboardViewModel =
                ViewModelProviders.of(this).get(TensorflowViewModel.class);
        View root = inflater.inflate(R.layout.fragment_tensorflow, container, false);
        final TextView textView = root.findViewById(R.id.text_tensorflow);
        dashboardViewModel.getText().observe(this, new Observer<String>() {
            @Override
            public void onChanged(@Nullable String s) {
                textView.setText(s);
            }
        });
        return root;
    }
}