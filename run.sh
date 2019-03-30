#!/bin/bash

# This file is used to run the different experiments

#echo "==================== START ==========================="
#echo "NLP"
#
#echo "Visualize data only"
#echo "python main.py --nlp \"minimal\" --skip_predict --data_visualization"
#python main.py --nlp "minimal" --skip_predict --data_visualization
#
#echo "python main.py --nlp \"minimal\" "
#python main.py --nlp "minimal"
#
#return_code=$?
#if (( $return_code == 0 )); then
#    echo "Success :) Exit code $return_code"
#    echo "********************************************************"
#    echo "python main.py --nlp \"generate_features\" --skip_predict --data_visualization"
#    python main.py --nlp "generate_features" --skip_predict --data_visualization
#
#    echo "python main.py --nlp \"minimal\" --parameter_tuning  --windows_mode \"one\" --evaluation_mode \"classifier\""
#    python main.py --nlp "minimal" --parameter_tuning  --windows_mode "one" --evaluation_mode "classifier"
#    echo "python main.py --nlp \"minimal\" --parameter_tuning  --windows_mode \"one\" --evaluation_mode \"major\""
#    python main.py --nlp "minimal" --parameter_tuning  --windows_mode "one" --evaluation_mode "major"
#    echo "python main.py --nlp \"minimal\" --parameter_tuning  --windows_mode \"one\" --evaluation_mode \"weighted\""
#    python main.py --nlp "minimal" --parameter_tuning  --windows_mode "one" --evaluation_mode "weighted"
#
#    echo "********************************************************"
#    echo "python main.py --nlp \"minimal\" --parameter_tuning  --windows_mode \"multiple\" --evaluation_mode \"classifier\""
#    python main.py --nlp "minimal" --parameter_tuning  --windows_mode "multiple" --evaluation_mode "classifier"
#    echo "python main.py --nlp \"minimal\" --parameter_tuning  --windows_mode \"multiple\" --evaluation_mode \"major\""
#    python main.py --nlp "minimal" --parameter_tuning  --windows_mode "multiple" --evaluation_mode "major"
#    echo "python main.py --nlp \"minimal\" --parameter_tuning  --windows_mode \"multiple\" --evaluation_mode \"weighted\""
#    python main.py --nlp "minimal" --parameter_tuning  --windows_mode "multiple" --evaluation_mode "weighted"
#
#    echo "********************************************************"
#    echo "python main.py --nlp \"minimal\" --parameter_tuning  --windows_mode \"sliding\" --evaluation_mode \"classifier\""
#    python main.py --nlp "minimal" --parameter_tuning  --windows_mode "sliding" --evaluation_mode "classifier"
#    echo "python main.py --nlp \"minimal\" --parameter_tuning  --windows_mode \"sliding\" --evaluation_mode \"major\""
#    python main.py --nlp "minimal" --parameter_tuning  --windows_mode "sliding" --evaluation_mode "major"
#    echo "python main.py --nlp \"minimal\" --parameter_tuning  --windows_mode \"sliding\" --evaluation_mode \"weighted\""
#    python main.py --nlp "minimal" --parameter_tuning  --windows_mode "sliding" --evaluation_mode "weighted"
#else
#	echo "FAILED :( Exit code $return_code"
#fi
#
#echo "========================================================"
#echo "Stock Data"
#
#echo "Visualize data only"
#echo "python main.py --skip_predict --data_visualization"
#python main.py --skip_predict --data_visualization
#
#echo "********************************************************"
#echo "python main.py --skip_predict --data_visualization --enable_vix"
#python main.py --skip_predict --data_visualization --enable_vix
#echo "********************************************************"
#
#echo "********************************************************"
#echo "python main.py --parameter_tuning --enable_vix --windows_mode \"one\" --evaluation_mode \"classifier\""
#python main.py --parameter_tuning --enable_vix --windows_mode "one" --evaluation_mode "classifier"
#echo "python main.py --parameter_tuning --enable_vix --windows_mode \"one\" --evaluation_mode \"major\""
#python main.py --parameter_tuning --enable_vix --windows_mode "one" --evaluation_mode "major"
#echo "python main.py --parameter_tuning --enable_vix --windows_mode \"one\" --evaluation_mode \"weighted\""
#python main.py --parameter_tuning --enable_vix --windows_mode "one" --evaluation_mode "weighted"
#
#echo "********************************************************"
#echo "python main.py --parameter_tuning --enable_vix --windows_mode \"multiple\" --evaluation_mode \"classifier\""
python main.py --parameter_tuning --enable_vix --windows_mode "multiple" --evaluation_mode "classifier"
#echo "python main.py --parameter_tuning --enable_vix --windows_mode \"multiple\" --evaluation_mode \"major\""
#python main.py --parameter_tuning --enable_vix --windows_mode "multiple" --evaluation_mode "major"
#echo "python main.py --parameter_tuning --enable_vix --windows_mode \"multiple\" --evaluation_mode \"weighted\""
#python main.py --parameter_tuning --enable_vix --windows_mode "multiple" --evaluation_mode "weighted"
#
#echo "********************************************************"
#echo "python main.py --parameter_tuning --enable_vix --windows_mode \"sliding\" --evaluation_mode \"classifier\""
#python main.py --parameter_tuning --enable_vix --windows_mode "sliding" --evaluation_mode "classifier"
#echo "python main.py --parameter_tuning --enable_vix --windows_mode \"sliding\" --evaluation_mode \"major\""
#python main.py --parameter_tuning --enable_vix --windows_mode "sliding" --evaluation_mode "major"
#echo "python main.py --parameter_tuning --enable_vix --windows_mode \"sliding\" --evaluation_mode \"weighted\""
#python main.py --parameter_tuning --enable_vix --windows_mode "sliding" --evaluation_mode "weighted"

#echo "==================== END =============================="


