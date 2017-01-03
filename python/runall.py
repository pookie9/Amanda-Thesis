import biexponetial_regress
import twocompartment_regress
import twocompartment_clearance_regress

def f_test(rss1,rss2):
    return ((rss1-rss2)/(5-4))/(rss2/(10-5))

if __name__=='__main__':
    print f_test(0.2316921040702,0.007850019068809312)
#    (bestc1,bestc2,bestk1,bestk2,best_mse)=biexponetial_regress.run_default()
#    print "Biexponential model: R1(0)=",bestc1,'R2(0)=',bestc2,'K1=',bestk1,'K2=',bestk2,' MSE=',best_mse
#    (bestc1,bestc2,bestk1,bestk2,best_mse)=twocompartment_regress.run_default()
#    print "Two-compartment model: R1(0)=",bestc1,'R2(0)=',bestc2,'K1=',bestk1,'K2=',bestk2,' MSE=',best_mse
#    (bestc1,bestc2,bestk1,bestk2,bestk3,best_mse2)=twocompartment_clearance_regress.run_default()
#    print "Two-compartment-clearance model: R1(0)=",bestc1,'R2(0)=',bestc2,'K1=',bestk1,'K2=',bestk2,'K3=',bestk3,' MSE=',best_mse2

#    print "F-test:",f_test(best_mse,best_mse2)
