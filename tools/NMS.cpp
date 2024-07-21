#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
float iouthr=0.5;//iou阈值
struct bbox
{
    float x1,y1,x2,y2;//1为左上，2为右下
    float score;
};
bool compare(bbox a,bbox b){//置信度排序大到小
    return a.score>b.score;
}
float get_iou(bbox c,bbox d){
    float w_in=min(c.x2,d.x2)-max(c.x1,d.x1);
    float h_in=min(c.y2,d.y2)-max(c.y1,d.y1);
    if(w_in<0||h_in<0){
        return 0;
    }else{
        float s_in=w_in*h_in;//交矩形S
        float sc=(c.y2-c.y1)*(c.x2-c.x1);
        float sd=(d.y2-d.y1)*(d.x2-d.x1);
        return s_in/(sc+sd-s_in);
    }
}
vector<bbox> NMS(vector<bbox> bboxes){
    vector<bbox> out;
    sort(bboxes.begin(),bboxes.end(),compare);
    for(vector<bbox>::iterator it=bboxes.begin();it!=bboxes.end();it++){
        for(auto outit=out.begin();outit!=out.end();outit++){
            if(get_iou(*it,*outit)>iouthr){
                bboxes.erase(it);
                continue;
            }
        }
        out.push_back(*it);

    }
    return out;
}

