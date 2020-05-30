package fr.esgi.faceid.utils;

import javafx.scene.control.ContextMenu;
import javafx.scene.control.MenuItem;
import javafx.scene.control.TextInputDialog;
import javafx.scene.layout.HBox;

import java.util.function.Consumer;

/**
 * Created by Botan on 5/28/2020. 11:09 PM
 **/
public class UI {

    public static void implementAvatarContextMenu(HBox root, Runnable onRemove, Consumer<String> onRename) {
        ContextMenu contextMenu = new ContextMenu();

        MenuItem item1 = new MenuItem("Rename");
        item1.setOnAction(event -> {
            TextInputDialog inputDialog = new TextInputDialog();
            inputDialog.setHeaderText("What's his name");

            inputDialog.showAndWait().ifPresent(onRename);
        });
        MenuItem item2 = new MenuItem("Delete");
        item2.setOnAction(event -> onRemove.run());

        contextMenu.getItems().addAll(item1, item2);

        root.setOnContextMenuRequested(e -> contextMenu.show(root, e.getScreenX(), e.getScreenY()));
    }

}
