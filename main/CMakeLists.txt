idf_component_register(SRCS "esp_test.cpp"
                    INCLUDE_DIRS "."
                    EMBED_FILES "model/classifier.tflite"
                    )
                    
set(MODEL_FILE "${CMAKE_CURRENT_LIST_DIR}/model/classifier.tflite")

target_add_binary_data(${COMPONENT_LIB} "${MODEL_FILE}" BINARY)
