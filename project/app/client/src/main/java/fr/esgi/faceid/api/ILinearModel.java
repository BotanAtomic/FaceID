package fr.esgi.faceid.api;

import com.sun.jna.Library;
import com.sun.jna.Pointer;

/**
 * Created by Botan on 5/22/2020. 4:49 PM
 **/
public interface ILinearModel extends Library {

    Pointer createModel(int size);

}
