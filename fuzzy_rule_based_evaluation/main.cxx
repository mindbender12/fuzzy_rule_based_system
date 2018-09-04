#include <cpp_headers.h>
#include <glm_headers.h>
#include <utils.h>

using namespace std;

int main(int argc, char** argv)
{
    //input arguments: ./exec rulenum dimension inmemfile outmemfile testfile 

    int rule_num = atoi(argv[1]);
    int input_dim_num = atoi(argv[2]);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    float temp_rule_matrix1[rule_num][input_dim_num];
    float temp_rule_matrix2[rule_num][input_dim_num];
    string line;
    vector<glm::vec2> temp_rule_matrix;
    vector<glm::vec3> testInputs;

    //Read the parameters of the trained fuzzy rule based system
    ///////////////////////////////////////////////////////////////////
    ifstream readoutFis;
    readoutFis.open(argv[4]);

    //Read outmfs
    getline(readoutFis, line);
    vector<float> outparamvals = split(line, ",");

    //Read inmfs
    ifstream readinFis;
    readinFis.open(argv[3]);

    while(!readinFis.eof())
    {
        getline(readinFis, line);

        if(line[0]!=NULL) //to deal with the last empty line, basically this helps to ignore it
        {
            vector<float> v = split(line, ",");

            if(v.size()>0)
            {
                temp_rule_matrix.push_back(glm::vec2(v[0],v[1]));
            }
        }
    }

    int ij=0;
    for(int qq=0;qq<input_dim_num;qq++)
    {
        for(int jj=0;jj<rule_num;jj++)
        {
            glm::vec2 v21 = temp_rule_matrix[ij++];
            temp_rule_matrix1[jj][qq] = v21.x; // sigma vals
            temp_rule_matrix2[jj][qq] = v21.y; // mean vals   
        }
    }

    temp_rule_matrix.clear();

    ///////////////////////////////////////////////
    //Fuzzy rule based inference system creation
    ///////////////////////////////////////////////
    Rule_Based_System rulebase;
    rulebase.num_rules = rule_num;
    rulebase.num_input_dim = input_dim_num;
    rulebase.fuzzy_system_type = "TSK";

    for(int qq=0;qq<rule_num;qq++)
    {
        Rules rule;
        rule.membership_func_type = "GMF";
        
        for(int jj=0;jj<input_dim_num;jj++)
        {
            Membership_func mm;
            mm.sigma = temp_rule_matrix1[qq][jj]; // sigma vals
            mm.mean = temp_rule_matrix2[qq][jj]; // mean vals
            rule.inputmfs.push_back(mm);
        }

        for(int jj=0;jj<=input_dim_num;jj++)
        {
            rule.out_params.push_back(outparamvals[jj]);
        }

        rulebase.rules.push_back(rule);
    }

    ///////////////////////////////////////////////////////////
    // read test input files
    ifstream readinTest;
    readinTest.open(argv[5]);

    while(!readinTest.eof())
    {
        getline(readinTest, line);

        if(line[0]!=NULL) //to deal with the last empty line, basically this helps to ignore it
        {
            vector<float> v = split(line, ",");

            if(v.size()>0)
            {
                testInputs.push_back(glm::vec3(v[0],v[1],v[2]));
            }
        }
    }

    //predict all the inputs
    vector<float> predicted_res;
    for(int i=0;i<testInputs.size();i++)
    {
        //Create feature vector
        Feature_vector fvector;
        fvector.feature_vec.push_back(testInputs[i].x);
        fvector.feature_vec.push_back(testInputs[i].y);
        fvector.feature_vec.push_back(testInputs[i].z);

        float val = evaluate_rulebase(rulebase,fvector);
        predicted_res.push_back(val);
    }

    ofstream outTestRes;
    outTestRes.open("../test_data/outPredicted.csv");

    for(int i=0;i<predicted_res.size();i++)    
        outTestRes<<predicted_res[i]<<endl;
    
    outTestRes.close();

    cout<<"finished prediction"<<endl;   

	return 0;
}
