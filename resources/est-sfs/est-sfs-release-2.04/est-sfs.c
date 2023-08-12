#include "routine-library.h"
#include<gsl/gsl_multimin.h>

/*
gcc -o est-sfs est-sfs.c routine-library.c -lm -lgsl -lgslcblas
*/

// Version History
// ---------------
// 6th January 2018:  2.01 First release.
// 12th March 2018:   2.02 Increased max_config to 10000
// 9th May 2018:      2.03 Use GSL routines for function maximization
// 3rd May 2022:      2.04 Increase size of vectors to allow handling of bigger data sets

#define version_number 2.04

#define max_n_alleles 10000
#define max_n_outgroup 3
#define max_config 1000000
#define max_n_tree 100
#define max_random_ml 1000  // Number of random ML starts
#define JK 0                // Substitution models
#define kimura 1
#define rate6 2            // Each symmetrical substitution type has its own rate
#define A 0
#define C 1
#define G 2
#define T 3
#define unknown -1         // Code for missing data, applies to outgroups only
#define infty -999999999.0

struct data_struct
{
   int n;
   int n_major;
   int b_major;
   int b_minor;
   int b_outgroup[max_n_outgroup];
   double p_anc_major;               // The posterior prob that the major allele is ancestral
   double ptree [max_n_tree];        // Posterior probability for each tree
};

struct data_struct data[max_config];

int n_disagree[max_n_alleles], n_agree_major[max_n_alleles], n_agree_minor[max_n_alleles];
int n_alleles, ind1, n_config, n_outgroup, model, trace_flag, verbose = 0, debug = 0;
double k_est[max_n_outgroup*2];                   // number of branches = 2*n_outgroup - 1
double kappa;                   // transition:transversion rate
double r6_parm[6];           // 6 symmetrical mutation rates:
// A<->T 0
// A<->C 1
// A<->G 2
// T<->C 3
// T<->G 4
// C<->G 5
int state_tab_3_outgroup[17][2];  // Table of 16 states for 3 outgroups, 2 internal nodes.


int read_data_element(FILE *fptr, int *ingroup_data_vec,
   int outgroup_data_vec[4][max_n_outgroup],  int alleles_to_output, int ind)
{
   int x1, stat, i, n_alleles;
   char ch;
   n_alleles = 0;
   for (i=0; i<4; i++)
   {
      stat = fscanf(fptr, "%d", &x1);
      if (stat == EOF) return -99;
//      printf("stat %d x1 %d\n", stat, x1);
      n_alleles += x1;
      if (i != 3)
      {
         stat = fscanf(fptr, "%c", &ch);
         if (stat == EOF) gabort("Unexpected EOF", 0);
//         printf("stat %d ch %c\n", stat, ch);
      }
      if (ind == -1)
      {
         ingroup_data_vec[i] = x1;
      }
      else
      {
         outgroup_data_vec[i][ind] = x1;
      }
   }
   if ((alleles_to_output == -1) || (alleles_to_output == n_alleles))
   {
      return n_alleles;
   }
   return -1;
}


#define max_mutations 3

int assign_observed_type(int n_alleles, int *ingroup, int outgroup[4][max_n_outgroup],
   int *n_config, int n_outgroup, int *totsites, int site_ind)
{
   int i, j, k, l, ind_major, ind_minor, n_minor_alleles, max_n_major, max_n_minor, found, bo,
      config_ind;
   max_n_major = -1;
   for (j=0; j<4; j++)
   {
      if (ingroup[j] > max_n_major)
      {
         max_n_major = ingroup[j];
         ind_major = j;
      }
   }
   if (max_n_major == -1) gabort("max_n_major unassigned", -1);
   max_n_minor = -1;
   for (j=0; j<4; j++)
   {
      if ((ingroup[j] > max_n_minor)&&(j != ind_major))
      {
         max_n_minor = ingroup[j];
         ind_minor = j;
      }
   }
//   if (debug) printf("max_n_major %d ind_major %d max_n_minor %d ind_minor %d\n",
//      max_n_major, ind_major, max_n_minor, ind_minor);
   if (max_n_major >= max_n_alleles)
   {
      printf("max_n_major = %d\n", max_n_major);
      gabort("assign_observed_type: max_n_major >= max_n_alleles\n", max_n_major);
   }
   (totsites[max_n_major])++;
   found = 0;
   for (i=0; i<*n_config; i++)
   {
      if ((data[i].n_major != max_n_major) || (data[i].b_major != ind_major) 
         || (data[i].b_minor != ind_minor))
      {
//         printf("No match config %d\n", i);
         continue;
      }
//      printf("Ingroup for site %d matches with config %d\n", site_ind, i);
      for (j=0; j<n_outgroup; j++)
      {
         bo = data[i].b_outgroup[j];
//         printf("config %d outgroup %d bo %d\n", i, j, bo);
         if (outgroup[bo][j] == 0)          // If it's a 1 then we've found the config
         {
            goto next_config;
         }
      }
      found = 1;
      config_ind = i;
      break;
next_config:;
   }
//   if (debug) printf("found %d\n", found);
//   monitorinput();
   if (found == 0)
   {
      config_ind = *n_config;
//      if (debug) printf("found %d config_ind %d\n", found, config_ind);
      if (config_ind >= max_config)
      {
         printf("config_ind %d\n", config_ind);
         gabort("assign_observed_type: config_ind >= max_config", 0);
      }
//      monitorinput();
      (data[config_ind].n)++;
      data[config_ind].n_major = max_n_major;
      data[config_ind].b_major = ind_major;
      data[config_ind].b_minor = ind_minor;
      for (i=0; i<n_outgroup; i++)
      {
         data[config_ind].b_outgroup[i] = -1;             // Default for missing data
         for (j=0; j<4; j++)
         {
            if (outgroup[j][i] == 1)
            {
               data[config_ind].b_outgroup[i] = j;
            }
            else if (outgroup[j][i] != 0)
            {
               printf("Processing site %d: ", site_ind);
               gabort("Outgroup state was not 0 or 1", outgroup[j][i]);
            }
         }
      }
      (*n_config)++;
   }
   else
   {
//      printf("found %d config_ind %d\n", found, config_ind);
//      monitorinput();
      (data[config_ind].n)++;
   }
   return(config_ind);
}


void dump_configs(int n_config, int n_outgroup)
{
   int i, j;
   for (i=0; i<n_config; i++)
   {
      printf("Config %d n %d n-major %d b-major %d b-minor %d ", i, data[i].n,
         data[i].n_major, data[i].b_major, data[i].b_minor);
      for (j=0; j<n_outgroup; j++)
      {
         printf("outgroup-%d %d ", j+1, data[i].b_outgroup[j]);
      }
      printf("\n");
   }
}


void compute_poisson_vec(double k, double poiss_p[max_mutations+1][max_n_outgroup*2], int ind)
{
   int i;
//   printf("compute_poisson_vec k %lf ind %d\n", k, ind);
   poiss_p[0][ind]  = exp(-k);
//   printf("poiss_p[0][%d] %f\n", ind, poiss_p[0][ind]);
   for (i=1; i<=max_mutations; i++)                  // up to 3 mutations considered
   {
      poiss_p[i][ind] = poiss_p[i-1][ind]*k/(double)i;
//      printf("poiss_p[%d][%d] %f\n", i, ind, poiss_p[i][ind]);
   }
//  monitorinput();
}


int transition_change(int b1, int b2)
{
   if (((b1 == A)&&(b2 == G))||((b1 == G)&&(b2 == A))||((b1 == C)&&(b2 == T))||
       ((b1 == T)&&(b2 == C))) return 1;
   return 0;
}


int r6_type(int b1, int b2)
{
   int x;
   if (((b1 == A)&&(b2 == T))||((b1 == T)&&(b2 == A)))
   {
      x = 0;
   }
   else if (((b1 == A)&&(b2 == C))||((b1 == C)&&(b2 == A)))
   {
      x = 1;
   }
   else if (((b1 == A)&&(b2 == G))||((b1 == G)&&(b2 == A)))
   {
      x = 2;
   }
   else if (((b1 == T)&&(b2 == C))||((b1 == C)&&(b2 == T)))
   {
      x = 3;
   }
   else if (((b1 == T)&&(b2 == G))||((b1 == G)&&(b2 == T)))
   {
      x = 4;
   }
   else if (((b1 == C)&&(b2 == G))||((b1 == G)&&(b2 == C)))
   {
      x = 5;
   }
   else
   {
      gabort("r6_type: unknown b1, b2 combination", 0);
   }
   return(x);
}


double compute_p_for_branch(double poiss_p[max_mutations+1][max_n_outgroup*2], int b1,
   int b2, int ind, int model, double kappa, double *r6_parm,
   double poiss_p_r6[5][max_n_outgroup*2], double r6_p[3][6][max_n_outgroup*2])
{
   int i, r6t;
   double p, pp0, pp1, pp2, denom;
//   printf("compute_lik_for_branch: b1 %d b2 %d ind %d\n", b1, b2, ind);
   if ((b1 == -1) || (b2 == -1))         // The branch contains missing data at either end
   {
//      printf("Missing data: b1 %d b2 %d, returning p = 1.0\n", b1, b2);
      return(1.0);
   }
//   monitorinput();
   if (model != rate6)
   {
      pp0 = poiss_p[0][ind];
      pp1 = poiss_p[1][ind];
      pp2 = poiss_p[2][ind];
//      for (i=0; i<=2; i++)
//      {
//         printf("poiss_p[%d][%d] %lf\n", i, ind, poiss_p[i][ind]);
//      }
   }
   if (model == JK)
   {
      if (b1 == b2)
      {
         p = pp0 + pp2*(1.0/3.0);
//         printf("pp0 %lf pp2*(1.0/3.0) %lf\n", pp0, pp2*(1.0/3.0));
//         monitorinput();
      }
      else
      {
         p = pp1*(1.0/3.0) + pp2*(2.0/9.0);
//         if (trace_flag) printf("pp1*(1.0/3.0) %lf pp2*(2.0/9.0) %lf\n", pp1*(1.0/3.0), pp2*(2.0/9.0));
//         monitorinput();
      }
   }
   else if (model == kimura)
   {
      denom = kappa*kappa + 4.0*kappa + 4.0;
      if (b1 == b2)
      {
         p = pp0 + pp2*(2.0 + kappa*kappa)/denom;
      }
      else if (transition_change(b1, b2))
      {
         p = pp1*kappa/(kappa + 2.0) + pp2*2.0/denom;
      }
      else    // transversion
      {
         p = pp1/(kappa + 2.0) + pp2*2.0*kappa/denom;
      }
   }
   else if (model == rate6)
   {
      if (b1 == b2)
      {
         p = poiss_p_r6[0][ind] + r6_p[0][0][ind];
//         printf("poiss_p_r6[0][ind] %lf r6_p[0][0][ind] %lf\n",
//            poiss_p_r6[0][ind], r6_p[0][0][ind]);
//         monitorinput();
      }
      else
      {
         r6t = r6_type(b1, b2);
         p = 2.0*r6_p[1][r6t][ind] + 2.0*r6_p[2][r6t][ind];
//         if (trace_flag) printf("2.0*r6_p[1][r6t][ind] %lf 2.0*r6_p[2][r6t][ind] %lf\n",
//            2.0*r6_p[1][r6t][ind], 2.0*r6_p[2][r6t][ind]);
//         monitorinput();
      }
   }
   else
   {
      gabort("Invalid substitution model", model);
   }
//   if (trace_flag) printf("compute_p_for_branch: returning p %lf\n", p);
//   monitorinput();
   return(p);
}


double prob_for_tree(int b1, int config, int n_outgroup, int tree_ind,
   double poiss_p[max_mutations+1][max_n_outgroup*2], int model, double kappa,
   double *r6_parm, double poiss_p_r6[5][max_n_outgroup*2],
   double r6_p[3][6][max_n_outgroup*2])
{
   int b2, b3, b4, b5, b6;
   double p, p0, p1, p2, p3, p4;
   if (n_outgroup == 1)
   {
      b2 = data[config].b_outgroup[0];
      p = compute_p_for_branch(poiss_p, b1, b2, 0, model, kappa, r6_parm, poiss_p_r6, r6_p);
   }
   else if (n_outgroup == 2)
   {
      b2 = tree_ind - 1;                   // The unknown node, which can be A, T, C or G
      b3 = data[config].b_outgroup[0];
      b4 = data[config].b_outgroup[1];
//      printf("prob_for_tree: b1 %d b2 %d b3 %d b4 %d\n", b1, b2, b3, b4);
      p0 = compute_p_for_branch(poiss_p, b2, b1, 0, model, kappa, r6_parm, poiss_p_r6,
              r6_p);
      p1 = compute_p_for_branch(poiss_p, b2, b3, 1, model, kappa, r6_parm, poiss_p_r6,
              r6_p);
      p2 = compute_p_for_branch(poiss_p, b2, b4, 2, model, kappa, r6_parm, poiss_p_r6,
              r6_p);
      p = p0*p1*p2;
//      monitorinput();
   }
   else if (n_outgroup == 3)
   {
      b2 = state_tab_3_outgroup[tree_ind][0];  // 1st internal node, which can be A,T,C or G
      b3 = data[config].b_outgroup[0];         // Outgroup 1
      b4 = state_tab_3_outgroup[tree_ind][1];  // 2nd internal node, which can be A,T,C or G
      b5 = data[config].b_outgroup[1];         // Outgroup 2
      b6 = data[config].b_outgroup[2];         // Outgroup 3
//      printf("prob_for_tree: b1 %d b2 %d b3 %d b4 %d b5 %d b6 %d\n", b1, b2, b3, b4, b5, b6);
      p0 = compute_p_for_branch(poiss_p, b2, b1, 0, model, kappa, r6_parm, poiss_p_r6,
              r6_p);
      p1 = compute_p_for_branch(poiss_p, b2, b3, 1, model, kappa, r6_parm, poiss_p_r6,
              r6_p);
      p2 = compute_p_for_branch(poiss_p, b4, b2, 2, model, kappa, r6_parm, poiss_p_r6,
              r6_p);
      p3 = compute_p_for_branch(poiss_p, b4, b5, 3, model, kappa, r6_parm, poiss_p_r6,
              r6_p);
      p4 = compute_p_for_branch(poiss_p, b4, b6, 4, model, kappa, r6_parm, poiss_p_r6,
              r6_p);
      p = p0*p1*p2*p3*p4;
//      printf("p0 %lf p1 %lf p2 %lf p3 %lf p4 %lf p %lf\n", p0, p1, p2, p3, p4, p);
//      monitorinput();
   }
   else gabort("prob_for_tree: invalid n_outgroup", n_outgroup);
//   if (trace_flag == 1) printf("prob_for_tree: p %lf\n", p);
   return (p);
}


double compute_lik_wrt_seg_type(int b1, int config, int n_tree,
   double poiss_p[max_mutations+1][max_n_outgroup*2], int model, double kappa,
   double *r6_parm, double poiss_p_r6[5][max_n_outgroup*2],
   double r6_p[3][6][max_n_outgroup*2], double *ptree)
{
   int i;
   double p, tot_p;
   tot_p = 0.0;
//   printf("compute_lik_wrt_seg_type: n_tree %d b1 %d\n", n_tree, b1);
//   monitorinput();
   for (i=1; i<=n_tree; i++)
   {
      p = prob_for_tree(b1, config, n_outgroup, i, poiss_p, model, kappa, r6_parm,
             poiss_p_r6, r6_p);
      tot_p += p;
      ptree[i] = p;
   }
   return(tot_p);
}


double compute_log_likelihood_wrt_sfs_p_fixed(
   double poiss_p[max_mutations+1][max_n_outgroup*2], double sfs_p, int n_tree,
   int model, double kappa, double *r6_parm, double poiss_p_r6[5][max_n_outgroup*2],
   double r6_p[3][6][max_n_outgroup*2], double *ptree)
{
   double totlogl, p1, p2, logl;
   int i, j;
//   printf("compute_log_likelihood_wrt_sfs_p_fixed\n");
   totlogl = 0.0;
   for (i=0; i<n_config; i++)
   {
      if (data[i].n_major != ind1) continue;
//      printf("ind1 %d\n", ind1);
//      printf("Config %d n %d n-major %d b-major %d b-minor %d ", i, data[i].n,
//         data[i].n_major, data[i].b_major, data[i].b_minor);
//      for (j=0; j<n_outgroup; j++)
//      {
//         printf("outgroup-%d %d ", j+1, data[i].b_outgroup[j]);
//      }
//      printf("\n");
      p1 = compute_lik_wrt_seg_type(data[i].b_major, i, n_tree, poiss_p, model, kappa,
         r6_parm, poiss_p_r6, r6_p, ptree);
      p2 = compute_lik_wrt_seg_type(data[i].b_minor, i, n_tree, poiss_p, model, kappa,
         r6_parm, poiss_p_r6, r6_p, ptree);
      if (data[i].b_major == data[i].b_outgroup[0])
      {
         logl = log(p1*sfs_p)*(double)data[i].n;
//         printf("Condition 1: p1 %lf\n", p1);
      }
      else
      {
         logl = log(p2*(1.0 - sfs_p))*(double)data[i].n;
//         printf("Condition 2: p2 %lf\n", p2);
      }
//      printf("p1 %lf p2 %lf sfs_p %lf logl %lf\n", p1, p2, sfs_p, logl);
      totlogl += logl;
//      monitorinput();
   }
//   printf("compute_log_likelihood_wrt_sfs_p_fixed totlogl %lf\n", totlogl);
//   monitorinput();
   return totlogl;
}


double compute_log_likelihood_wrt_sfs_p_seg(double poiss_p[max_mutations+1][max_n_outgroup*2], double sfs_p, int n_tree, int model, double kappa, double *r6_parm,
   double poiss_p_r6[5][max_n_outgroup*2], double r6_p[3][6][max_n_outgroup*2],
   double *ptree, int use_sfs_p_as_prior)
{
   double totlogl, p1, p2, logl;
   int i, j;
   totlogl = 0.0;
   for (i=0; i<n_config; i++)
   {
      if (data[i].n_major != ind1)
      {
         continue;
      }
      p1 = compute_lik_wrt_seg_type(data[i].b_major, i, n_tree, poiss_p, model, kappa,
         r6_parm, poiss_p_r6, r6_p, ptree);
      if (use_sfs_p_as_prior) p1 *= sfs_p;
      p2 = compute_lik_wrt_seg_type(data[i].b_minor, i, n_tree, poiss_p, model, kappa,
         r6_parm, poiss_p_r6, r6_p, ptree);
      if (use_sfs_p_as_prior) p2 *= (1.0 - sfs_p);
      logl = log(p1*sfs_p + p2*(1.0 - sfs_p))*(double)data[i].n;
//      printf("p1 %lf p2 %lf sfs_p %lf logl %lf\n", p1, p2, sfs_p, logl);
      totlogl += logl;
//      monitorinput();
      data[i].p_anc_major = p1/(p1 + p2);
//      printf("Config %d p %lf\n", i, p1/(p1 + p2));
//      monitorinput();
   }
   return totlogl;
}


double lookuprate(int b1, int b2, double *r6_parm)
{
   if (((b1==A)&&(b2==T))||((b1==T)&&(b2==A))) return(r6_parm[0]);
   else if (((b1==A)&&(b2==C))||((b1==C)&&(b2==A))) return(r6_parm[1]);
   else if (((b1==A)&&(b2==G))||((b1==G)&&(b2==A))) return(r6_parm[2]);
   else if (((b1==T)&&(b2==C))||((b1==C)&&(b2==T))) return(r6_parm[3]);
   else if (((b1==T)&&(b2==G))||((b1==G)&&(b2==T))) return(r6_parm[4]);
   else if (((b1==C)&&(b2==G))||((b1==G)&&(b2==C))) return(r6_parm[5]);
   else gabort("lookuprate: unknown b1 b2 combination", 0);
}



// k1 is the total rate of base X -> Y, k2 is the total rate of base Y -> X
double intfunc(double k1, double k2)
{
   double num, denom, diff;
   diff = k1 - k2;
   if (diff < 0.0) diff = -diff;
   if (diff <0.000001)
   {
      num = k1*k1*exp(-k1);
      denom = 2.0;
   }
   else
   {
      num = k1*k2*exp(-k2-k1)*(exp(k2) - exp(k1)*k2 + (k1 - 1.0)*exp(k1));
      denom = k2*k2 - 2.0*k1*k2 + k1*k1;
   }
//   printf("intfunc: k1 %lf k2 %lf num %lf denom %lf\n", k1, k2, num, denom);
   return(num/denom);
}



double compute_poiss_r6_p2_no_change(double k, double *r6_parm)
{
   int i, j;
   double totp, k1, p, y, totkvec[4], poiss_p_r6[5][max_mutations+1], res;
   for (i=0; i<=3; i++) totkvec[i] = 0.0;
   for (i=0; i<=3; i++)
   {
      for (j=0; j<=3; j++)
      {
         if (i == j) continue; // Only consider probability of changes   }
         y = lookuprate(i, j, r6_parm)*k*2.0;
         totkvec[i] += y;
      }
//      printf("Base %c totvec[%d] %lf\n", basechar(i), i, totkvec[i]);
   }
//   monitorinput();
   res = 0.0;
   for (i=0; i<=3; i++)   // four bases
   {
      totp = 0.0;
      for (j=0; j<=3; j++) // four bases
      {
         if (i == j) continue; // Only consider probability of changes
         k1 = lookuprate(i, j, r6_parm)*k*2.0;
         if (k1 == 0.0) continue;
         p = intfunc(totkvec[i], totkvec[j]);
         p /= totkvec[i]/k1;
         p /= totkvec[j]/k1;
         totp += p;
//         printf("i %c j %c k1 %lf totkvec[%d] %lf totkvec[%d] %lf p %lf\n",
//            basechar(i),basechar(j), k1, i, totkvec[i], j, totkvec[j], p);
      }
//      printf("Rate change %c <-> %c = %lf\n", basechar(i), basechar(i), totp);
//      monitorinput();
      poiss_p_r6[i+1][2] = totp;
      res += totp;
   }
   res /= 4.0;
//   printf("compute_poiss_r6_p2_no_change res %lf\n", res);
   return(res);
}


double r6_p1_contrib(int ind, double r[2][3], double k)
{
   int i;
   double kr6, p, res;
   kr6 = 0.0;
   for (i=0; i<3; i++)
   {
      kr6 += r[ind][i];
   }
   if (r[ind][0] == 0.0) return(0.0);
   p = r[ind][0]/kr6;
   kr6 *= 2.0*k;
//   printf("r6_p1_contrib: ind %d p %lf kr6 %lf\n", ind, p, kr6);
   res = exp(-kr6)*kr6*p;
   return(res);
}


double compute_r6_p1(int r6_ind, double *r6_parm, double k)
{
   static double r[2][3], res;
   double c0, c1;
   r[0][0] = r6_parm[r6_ind];
   r[1][0] = r6_parm[r6_ind];
   if (r6_ind == 0) // A<->T
   {
      r[0][1] = r6_parm[1];
      r[0][2] = r6_parm[2];
      r[1][1] = r6_parm[3];
      r[1][2] = r6_parm[4];
   }
   else if (r6_ind == 1) // A<->C
   {
      r[0][1] = r6_parm[0];
      r[0][2] = r6_parm[2];
      r[1][1] = r6_parm[3];
      r[1][2] = r6_parm[5];
   }
   else if (r6_ind == 2) // A<->G
   {
      r[0][1] = r6_parm[0];
      r[0][2] = r6_parm[1];
      r[1][1] = r6_parm[4];
      r[1][2] = r6_parm[5];
   }
   else if (r6_ind == 3) // T<->C
   {
      r[0][1] = r6_parm[0];
      r[0][2] = r6_parm[4];
      r[1][1] = r6_parm[1];
      r[1][2] = r6_parm[5];
   }
   else if (r6_ind == 4) // T<->G
   {
      r[0][1] = r6_parm[0];
      r[0][2] = r6_parm[3];
      r[1][1] = r6_parm[2];
      r[1][2] = r6_parm[5];
   }
   else if (r6_ind == 5) // C<->G
   {
      r[0][1] = r6_parm[1];
      r[0][2] = r6_parm[3];
      r[1][1] = r6_parm[2];
      r[1][2] = r6_parm[4];
   }
   else
   {
      gabort("compute_r6_p1: unknown r6_ind", r6_ind);
   }
   c0 = r6_p1_contrib(0, r, k);
   c1 = r6_p1_contrib(1, r, k);
//   printf("compute_r6_p1: r6_ind %d c0 %lf c1 %lf\n", r6_ind, c0, c1);
   res = (c0 + c1)/4.0;
   return (res);
}

void look_up_bases(int r6_ind, int *b1, int *b2)
{
   if (r6_ind == 0)
   {
      *b1 = A;
      *b2 = T;
   }
   else if (r6_ind == 1)
   {
      *b1 = A;
      *b2 = C;
   }
   else if (r6_ind == 2)
   {
      *b1 = A;
      *b2 = G;
   }
   else if (r6_ind == 3)
   {
      *b1 = T;
      *b2 = C;
   }
   else if (r6_ind == 4)
   {
      *b1 = T;
      *b2 = G;
   }
   else if (r6_ind == 5)
   {
      *b1 = C;
      *b2 = G;
   }
   else gabort("look_up_bases: unknown r6_ind", r6_ind);
}

double p_for_two_changes(int b1, int b2, double *r6_parm, double k)
{
   int i, j;
   double y, p1, p2, totp, k1, k2, p, totkvec[4];
   for (i=0; i<=3; i++) totkvec[i] = 0.0;
   for (i=0; i<=3; i++)
   {
      for (j=0; j<=3; j++)
      {
         if (i == j) continue; // Only consider probability of changes
         y = lookuprate(i, j, r6_parm)*k*2.0;
         totkvec[i] += y;
      }
//      printf("Base %c totvec[%d] %lf\n", basechar(i), i, totkvec[i]);
   }
//   monitorinput();
   totp = 0.0;
   for (i=0; i<=3; i++)   // four bases
   {
      if ((i == b1)||(i == b2)) continue;
      k1 = lookuprate(b1, i, r6_parm)*k*2.0;
      if (k1 == 0.0) continue;
      k2 = lookuprate(i, b2, r6_parm)*k*2.0;
      if (k2 == 0.0) continue;
      p = intfunc(totkvec[b1], totkvec[i]);
      p /= totkvec[b1]/k1;
      p /= totkvec[i]/k2;
      totp += p;
//    printf("b1 %c i %c k1 %lf k2 %lf totkvec[%d] %lf totkvec[%d] %lf p %lf\n",
//         basechar(b1),basechar(i), k1, k2, b1, totkvec[b1], i, totkvec[i], p);
   }
//   printf("Rate change %c <-> %c = %lf\n", basechar(b1), basechar(b2), totp);
   return(totp);
}



double compute_poiss_r6_p2_change(int r6_ind, double *r6_parm, double k)
{
   double p1, p2;
   int b1, b2;
   look_up_bases(r6_ind, &b1, &b2);
   p1 = p_for_two_changes(b1, b2, r6_parm, k);
   p2 = p_for_two_changes(b2, b1, r6_parm, k);
   return(p1 + p2)/4.0;
}



void compute_r6_probs(int n_branch, double *k_est, double poiss_p_r6[5][max_n_outgroup*2],
   double *r6_parm, double r6_p[3][6][max_n_outgroup*2])
{
   int i, j;
   double k;
   for (i=0; i<n_branch; i++)
   {
      k = k_est[i];
      poiss_p_r6[A][i] = exp(-k*(r6_parm[0] + r6_parm[1] + r6_parm[2])*2.0);
      poiss_p_r6[T][i] = exp(-k*(r6_parm[0] + r6_parm[3] + r6_parm[4])*2.0);
      poiss_p_r6[C][i] = exp(-k*(r6_parm[1] + r6_parm[3] + r6_parm[5])*2.0);
      poiss_p_r6[G][i] = exp(-k*(r6_parm[2] + r6_parm[4] + r6_parm[5])*2.0);
      poiss_p_r6[0][i] = (poiss_p_r6[A][i] + poiss_p_r6[C][i] + 
                          poiss_p_r6[G][i] + poiss_p_r6[T][i])/4.0;
// Note, prob. of 2 changes with no substitution goes in index 0
      r6_p[0][0][i] = compute_poiss_r6_p2_no_change(k, r6_parm);
      for (j=0; j<6; j++)
      {
// Note, prob. of 1 change with one substitution goes in index 1
         r6_p[1][j][i] = compute_r6_p1(j, r6_parm, k);
// Note, prob. of 2 changes with one substitution goes in index 2
         r6_p[2][j][i] = compute_poiss_r6_p2_change(j, r6_parm, k);
      }
   }
}


double compute_log_likelihood_wrt_sfs_p_gsl(const gsl_vector *v, void *params)
{
   int i, n_tree, n_branch;
   static double poiss_p[max_mutations+1][max_n_outgroup*2],
      poiss_p_r6[5][max_n_outgroup*2], r6_p[3][6][max_n_outgroup*2];
   static double ptree[max_n_tree];
   double log_l, sfs_p;
   sfs_p = gsl_vector_get(v, 0);

   if ((sfs_p <= 0.0)||(sfs_p >= 1.0))
   {
      if (verbose) printf("sfs_p %lf out of range - returning %lf\n", sfs_p, -infty);
//     monitorinput();
      return(-infty);
   }
//   printf("compute_log_likelihood_wrt_sfs_p: ind1 %d\n", ind1);
   for (i=0; i<2*n_outgroup - 1; i++)              // Number of branches
   {
      compute_poisson_vec(k_est[i], poiss_p, i);
   }
   n_tree = pow(4, n_outgroup -1);
   if (n_tree >= max_n_tree)
   {
      printf("n_tree (%d) exceeds max_n_tree (%d). Terminating\n", n_tree, max_n_tree);
      gabort("Program aborting2", 0);
   }
//   printf("n_tree %d n_outgroup %d\n", n_tree, n_outgroup);
//   monitorinput();
   if (model == rate6)
   {
      n_branch = 2*n_outgroup - 1;
      compute_r6_probs(n_branch, k_est, poiss_p_r6, r6_parm, r6_p);
   }
   if (ind1 == n_alleles)
   {
      log_l = compute_log_likelihood_wrt_sfs_p_fixed(poiss_p, sfs_p, n_tree, model, kappa,
         r6_parm, poiss_p_r6, r6_p, ptree);
   }
   else
   {
      log_l = compute_log_likelihood_wrt_sfs_p_seg(poiss_p, sfs_p, n_tree, model, kappa,
         r6_parm, poiss_p_r6, r6_p, ptree, 0);
//      printf("ind1 %d sfs_p %lf log_l %lf\n", ind1, sfs_p, log_l);
//      monitorinput();
   }
   return(-log_l);
}


double compute_log_likelihood_wrt_k_kappa_rate6_fixed(
   double poiss_p[max_mutations+1][max_n_outgroup*2], int n_tree, int model, double kappa,
   double *r6_parm, double poiss_p_r6[5][max_n_outgroup*2],
   double r6_p[3][6][max_n_outgroup*2])
{
   double totlogl, p1, logl;
   static double ptree[max_n_tree];
   int i, j;
//   printf("compute_log_likelihood_wrt_sfs_p_fixed n_config %d\n", n_config);
   totlogl = 0.0;
   for (i=0; i<n_config; i++)
   {
      if (data[i].n_major != n_alleles) continue;
//      printf("Config %d n %d n-major %d b-major %d b-minor %d ", i, data[i].n,
//         data[i].n_major, data[i].b_major, data[i].b_minor);
//      for (j=0; j<n_outgroup; j++)
//      {
//         printf("outgroup-%d %d ", j+1, data[i].b_outgroup[j]);
//      }
//      printf("\n");
      p1 = compute_lik_wrt_seg_type(data[i].b_major, i, n_tree, poiss_p, model, kappa,
         r6_parm, poiss_p_r6, r6_p, ptree);
      logl = log(p1)*(double)data[i].n;
//      printf("compute_log_likelihood_wrt_k_kappa_rate6_fixed: p1 %lf logl %lf\n", p1, logl);
      totlogl += logl;
      for (j=1; j<=n_tree; j++)
      {
         data[i].ptree[j] = ptree[j];
      }
//      monitorinput();
   }
   return totlogl;
}


double compute_log_likelihood_wrt_k_kappa_rate6_seg(int ind,
   double poiss_p[max_mutations+1][max_n_outgroup*2],
   int n_tree, int model, double kappa, double *r6_parm,
   double poiss_p_r6[5][max_n_outgroup*2], double r6_p[3][6][max_n_outgroup*2])
{
   double totlogl, p1, p2, logl;
   static double ptree1[max_n_tree], ptree2[max_n_tree];
   int i, j;
   totlogl = 0.0;
   for (i=0; i<n_config; i++)
   {
      if (data[i].n_major != ind) continue;
      p1 = compute_lik_wrt_seg_type(data[i].b_major, i, n_tree, poiss_p, model, kappa,
         r6_parm, poiss_p_r6, r6_p, ptree1);
      p2 = compute_lik_wrt_seg_type(data[i].b_minor, i, n_tree, poiss_p, model, kappa,
         r6_parm, poiss_p_r6, r6_p, ptree2);
//      logl = log(p1 + p2)*(double)data[i].n;
      logl = log(0.5*(p1 + p2))*(double)data[i].n; // Take the average probability. This
                                       // 0.5*actually makes no difference to the outcome.
//      printf("compute_log_likelihood_wrt_k_kappa_rate6_seg: p1 %lf p2 %lf logl %lf\n", p1, p2, logl);
      totlogl += logl;
      for (j=1; j<=n_tree; j++)
      {
         data[i].ptree[j] = (ptree1[j] + ptree2[j])/2.0;  // Take the average probability
      }
//      monitorinput();
   }
   return totlogl;
}



char basechar(int b)
{
   if (b == 0) return('A');
   else if (b == 1) return('C');
   else if (b == 2) return('G');
   else if (b == 3) return('T');
   else gabort("basechar: unknown base", b);
}



double compute_log_likelihood_wrt_k_kappa_rate6_gsl(const gsl_vector *v, void *params)
{
   int i, j, n_tree, n_branch;
   static double poiss_p[max_mutations+1][max_n_outgroup*2],
      poiss_p_r6[5][max_n_outgroup*2], r6_p[3][6][max_n_outgroup*2];
   double log_l, tot_log_l, sum, r6, k;
   n_branch = 2*n_outgroup - 1;
   for (i=0; i<n_branch; i++)
   {
      k_est[i] = gsl_vector_get(v, i);
      if (k_est[i] <= 0.0)
      {
         if (verbose) printf("k_est[%d] %lf - returning %lf\n", i, k_est[i], -infty);
//        monitorinput();
         return(-infty);
      }
      compute_poisson_vec(k_est[i], poiss_p, i);
   }
   if (model == kimura)
   {
      kappa = gsl_vector_get(v, n_branch);
      if (kappa <= 0.0)
      {
          if (verbose) printf("kappa %lf - returning %lf\n", kappa, -infty);
//        monitorinput();
         return(-infty);
      }
   }
   else if (model == rate6)
   {
      sum = 0.0;
      for (i=n_branch; i<n_branch+5; i++)
      {
         r6_parm[i-n_branch] = gsl_vector_get(v, i);
         r6 = r6_parm[i-n_branch];
//         printf("r6_parm[%d] %lf\n", i-n_branch, r6_parm[i-n_branch]);
         if ((r6 <= 0.0)||(r6 >= 1.0))
         {
            if (verbose) printf("r6_parm[%d] = %lf - returning %lf\n", i-n_branch, r6, -infty);
//            monitorinput();
            return(-infty);
         }
         sum += r6;
      }
      if (sum >= 1.0)
      {
         if (verbose) printf("sum(r6_parm) = %lf - returning %lf\n", sum, -infty);
//         monitorinput();
         return(-infty);
      }
      r6_parm[5] = 1.0 - sum;
//      printf("r6_parm[%d] %lf\n", 5, r6_parm[5]);
      compute_r6_probs(n_branch, k_est, poiss_p_r6, r6_parm, r6_p);
   }
//   monitorinput();
   n_tree = pow(4, n_outgroup - 1);
   tot_log_l = compute_log_likelihood_wrt_k_kappa_rate6_fixed(poiss_p, n_tree, model,
      kappa, r6_parm, poiss_p_r6, r6_p);
//   printf("compute_log_likelihood_wrt_k_kappa_fixed: logl %lf\n", tot_log_l);
   for (i=1; i<n_alleles; i++)
   {
      log_l = compute_log_likelihood_wrt_k_kappa_rate6_seg(i, poiss_p, n_tree, model,
         kappa, r6_parm, poiss_p_r6, r6_p);
//      printf("Segregating classes: i %d log_l %lf\n", i, log_l);
      tot_log_l += log_l;
   }
   for (i=0; i<n_branch; i++)
   {
      if (verbose) printf("k_est[%d] %lf ", i, k_est[i]);
   }
   if (model == rate6)
   {
      if (verbose) printf("r6: ");
      for (i=0; i<=5; i++)
      {
         if (verbose) printf("%lf ", r6_parm[i]);
      }
   }
   if (model == kimura)
   {
      if (verbose) printf("kappa %lf ", kappa);
   }
   if (verbose) printf("tot_log_l %lf\n", tot_log_l);
//   monitorinput();
   return(-tot_log_l);
}


double dolik_wrt_k_kappa_rate6_gsl(double *k_est, double kappa, int n_branch, double *r6_parm)
{
   int np, i, nevals;
   double stepsize = 0.05, min, convcrit = 1.0e-10;

/* The parameter vector for the function defn: UNUSED */
   double par[2] = {1.0, 2.0};

   gsl_multimin_fminimizer *s;
   gsl_vector *ss, *x;
   gsl_multimin_function ex4_fn;
   size_t iter = 0;
   int status;
   double size;

   nevals = 1000;
   np = n_branch;
   if (model==kimura)
   {
      np++;         // For parameter kappa
   }
   else if (model==rate6)
   {
      np = np+5; // 6 possible symmetrical changes = 5df
   }
//   printf("np %d\n", np);
/* Initial vertex size vector */
   ss = gsl_vector_alloc (np);
/* Set all step sizes to stepsize */
   gsl_vector_set_all (ss, stepsize);
/* Starting point */
   x = gsl_vector_alloc (np);
//   printf("n_branch %d\n", n_branch);
   for (i = 0; i<n_branch; i++)
   {
//      printf("Initialize branch parameter %d\n", i);
      gsl_vector_set (x, i, k_est[i]);      // Branch length starting value
   }
   if (model==kimura)
   {
//      printf("Initialize parameter %d\n", n_branch);
      gsl_vector_set (x, n_branch, kappa);     // Kappa starting value
   }
   else if (model==rate6)
   {
      for (i=n_branch; i<n_branch+5; i++)
      { 
//         printf("Initialize R6 parameter %d\n", i);
         gsl_vector_set (x, i, r6_parm[i-n_branch]);      // R6 starting value
      }
   }
/* Initialize method and iterate */
   ex4_fn.f = &compute_log_likelihood_wrt_k_kappa_rate6_gsl;
   ex4_fn.n = np;
   ex4_fn.params = (void *)&par;
   s = gsl_multimin_fminimizer_alloc (gsl_multimin_fminimizer_nmsimplex , np);
   gsl_multimin_fminimizer_set (s, &ex4_fn, x, ss);

//   printf("Done initialize method\n");
//   monitorinput();
   do
   {
      iter++;
      status = gsl_multimin_fminimizer_iterate(s);
      if (status)
      break;
      size = gsl_multimin_fminimizer_size (s);
      status = gsl_multimin_test_size (size, convcrit);
      if (status == GSL_SUCCESS)
      {
//         printf ("converged to minimum at\n");
      }
//      printf ("%5d ", iter);
      for (i = 0; i < np; i++)
      {
//         printf ("%10.3e ", gsl_vector_get (s->x, i));
      }
//      printf ("f() = %7.3f size = %.3f\n", s->fval, size);
//      monitorinput();
   }
   while (status == GSL_CONTINUE && iter < nevals);
   for (i=0; i<n_branch; i++)
   {
      k_est[i] = gsl_vector_get(s->x, i);
      if (verbose) printf("ML estimates k[%d] %lf\n", i, k_est[i]);
   }
   if (model==kimura)
   {
      kappa = gsl_vector_get(s->x, n_branch);
      if (verbose) printf("ML estimate kappa %lf\n", kappa);
   }
   else if (model == rate6)
   {
      for (i=n_branch; i<n_branch+5; i++)
      {
         r6_parm[i-n_branch] = gsl_vector_get(s->x, i);
         if (verbose) printf("ML estimates r6[%d] %lf\n", i-n_branch, r6_parm[i-n_branch]);
      }
      if (verbose) printf("ML estimates r6[5] %lf\n", r6_parm[5]);
   }
   min = -s->fval;
   if (verbose) printf("ML %lf\n", min);
   gsl_vector_free(x);
   gsl_vector_free(ss);
   gsl_multimin_fminimizer_free (s);
//   monitorinput();
   return(min);
}


void get_int_parm(char *str, int *parm, FILE *fptr)
{
   int stat;
   char x[100];
   stat = fscanf(fptr, "%s %d", x, parm);
   if (stat != 2) gabort("Read error 1 get_int_parm", 0);
   stat = strcmp(x, str);
   if (stat != 0)
   {
      printf("Processing %s: ", str);
      gabort("failed to find string in config file", 0);
   }
}


void get_double_parm(char *str, double *parm, FILE *fptr)
{
   int stat;
   char x[100];
   stat = fscanf(fptr, "%s %lf", x, parm);
   if (stat != 2) gabort("Read error 1 get_int_parm", 0);
   stat = strcmp(x, str);
   if (stat != 0)
   {
      printf("Processing %s: ", str);
      gabort("failed to find string in config file", 0);
   }
}


void parse_config_file(char *config_file, int *n_outgroup, double *k_est, int *model, 
   double *kappa, double *r6_parm, int *nrandom)
{
   FILE *fptr;
   char x[100], y[100];
   int stat, i, n_branch;
   double sum = 0;
   fptr = openforreadsilent(config_file);
   get_int_parm("n_outgroup", n_outgroup, fptr);
   n_branch = 2*(*n_outgroup) - 1;
   get_int_parm("model", model, fptr);
   if ((*model < 0)||(*model > 2)) gabort("Model must be >=0 and <=2", *model);
   get_int_parm("nrandom", nrandom, fptr);
   if (*nrandom < 0) gabort("nrandom must be an integer", *nrandom);
   if (*nrandom > max_random_ml)  gabort("nrandomm too large. Maximum = ", max_random_ml);
   if (*nrandom == 0)  // Don't generate random starting values: assign them
   {
      for (i=0; i<n_branch; i++)
      {
         k_est[i] = 0.05;
      }
      if (*model == kimura)
      {
         *kappa = 2.0;   
      }
      else if (*model == rate6)
      {
         for (i=0; i<5; i++)               // Number of parameters = 6
         {
            r6_parm[i] = 1.0/6.0;
            sum += r6_parm[i];
         }
      }
      if (sum > 0.9999999) gabort("Sum of rate6 starting values 0..4 >  0.9999999", 0);
      else r6_parm[5] = 1.0 - sum;
   }
   fclose(fptr);
}


void set_up_state_tab_3_outgroup(int state_tab_3_outgroup[17][2])
{
   int i, b1, b2;
   b1 = 0; b2 = 0;
   for (i=1; i<=16; i++)
   {
      state_tab_3_outgroup[i][0] = b1;
      state_tab_3_outgroup[i][1] = b2;
      b1++;
      if (b1 > 3) b1 = 0;
      if (b1 == 0) b2++;
//      printf("state_tab_3_outgroup[%d][0] %d  state_tab_3_outgroup[%d][1] %d\n",
//         i, state_tab_3_outgroup[i][0], i, state_tab_3_outgroup[i][1]);
   }
}


void set_up_random_starting_values(int n_branch, double *k_est, int model, double *kappa,
   double *r6_parm)
{
   int i;
   double sum = 0.0, u;
   u = uniform();
//   printf("set_up_random_starting_values u %lf\n", u);
   for (i=0; i<n_branch; i++)
   {
      k_est[i] = uniform()*0.2;
   }
   if (model == kimura)
   {
      *kappa = uniform()*10.0;
   }
   else if (model == rate6)
   {
      for (i=0; i<=5; i++)
      {
         r6_parm[i] = uniform();
         sum += r6_parm[i];
      }
      for (i=0; i<=5; i++)
      {
         r6_parm[i] /= sum;
      }
   }
}


void store_ml_estimates(int n_branch, double *k_est, double *k_est_saved, int model,
   double kappa, double *kappa_saved, double *r6_parm, double *r6_parm_saved)
{
   int i;
   for (i=0; i<n_branch; i++)
   {
      k_est_saved[i] = k_est[i];
   }
   if (model == kimura) *kappa_saved = kappa;
   else if (model == rate6)
   {
      for (i=0; i<=5; i++)
      {
         r6_parm_saved[i] = r6_parm[i];
      }
   }
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////

void main(int argc, char *argv[])
{
   FILE *fptr, *outfile_ptr;
   static char data_file[1000], output_file[1000], config_file[1000], output_file_p_anc[1000],
      seedfile[1000];
   static int ingroup[4], outgroup[4][max_n_outgroup];
   int nsites, i, j, stat, prev_n_alleles = -1, alleles_to_output, start, sites_in_sfs_cat;
   double log_l, sfs_p_est, p, maxlogl, ptot, rdum, current_ml = 0;
   static double sfs[max_n_alleles+1], sfs_with_prior[max_n_alleles+1], sfs_p_est_vec[max_n_alleles+1];
   double tol = 0.00000001, res, xmin, kappa_saved, prev_log_l, diff_log_l;
   static double k_est_saved[max_n_outgroup*2], r6_parm_saved[6], random_start_ml[max_random_ml];
   int totsites[max_n_alleles+1], n_branch, config_ind, sind, n_tree, nrandom, reps_with_random_start;
   static double poiss_p[max_mutations+1][max_n_outgroup*2],
      poiss_p_r6[5][max_n_outgroup*2], r6_p[3][6][max_n_outgroup*2];

// For GSL simplex maximization of SFS elements
   int np, nevals;
   double stepsize = 0.05, min, convcrit = 1.0e-10;

/* The parameter vector for the function defn: UNUSED */
   double par[2] = {1.0, 2.0};

   gsl_multimin_fminimizer *s;
   gsl_vector *ss, *x;
   gsl_multimin_function ex4_fn;
   size_t iter = 0;
   int status;
   double size;

   for (i=0; i<max_config; i++) data[i].p_anc_major = 1.0; // Default for fixed sites
//   printf("argc %d\n", argc);
//   printf("Input file is assumed to have the same number of alleles at each site\n");
   alleles_to_output = -1;
   if ((argc != 6)&&(argc != 7))
   {
      printf("USAGE: $ %s config_file_name input_file_name seedfile_name output_file_sfs output_file_p_anc <OPTIONAL verbose>\n", argv[0]);
      gabort("argc = ", argc);
   }
   printf("%s version %2.2f\n", argv[0], version_number);
//   monitorinput();
   strcpy(config_file, argv[1]);
   strcpy(data_file, argv[2]);
   strcpy(seedfile, argv[3]);
   strcpy(output_file, argv[4]);
   strcpy(output_file_p_anc, argv[5]);
   if (argc == 7)
   {
      if (strcmp("verbose", argv[6]) == 0)
      {
         verbose = 1;
      }
   }
//   printf("verbose %d\n", verbose);
   if (verbose) 
   {
      printf("config_file %s data_file %s seedfile %s output_file_sfs %s output_file_p_anc %s verbose %d\n", config_file, data_file, seedfile, output_file, output_file_p_anc, verbose);
      printf("alleles_to_output %d\n", alleles_to_output);
   }
   for (i=0; i<=max_n_alleles; i++) totsites[i] = 0;

   parse_config_file(config_file, &n_outgroup, k_est, &model, &kappa, r6_parm, &nrandom);
   n_branch = 2*n_outgroup - 1;
   if (verbose) 
   {
      printf("n_outgroup %d model %d n_branch %d nrandom %d",
         n_outgroup, model, n_branch, nrandom);
      printf("\n");
   }
   if (nrandom == 0)
   {
      for (i=0; i<n_branch; i++)               // Number of branches
      {
         if (verbose) printf("Starting values: k_est[i] %lf\n", k_est[i]);
      }
      if (model == kimura)
      {
         if (verbose) printf("kappa %lf\n", kappa);
      }
      else if (model == rate6)
      {
         for (i=0; i<=5; i++)               // Number of parameters = 6
         {
            if (verbose) printf("r6_parm[%d] %lf\n", i, r6_parm[i]);
         }
      }
   }
   getseedquick(seedfile);

   if (n_outgroup == 3) set_up_state_tab_3_outgroup(state_tab_3_outgroup);
//   monitorinput();
   fptr = openforreadsilent(data_file);
   n_config = 0;
   nsites = 0;
   int counter = 0;
   for (;;)
   {
      stat = read_data_element(fptr, ingroup, outgroup, alleles_to_output, -1);
      if (stat==-99) break;
      if (stat == -1) continue;
      n_alleles = stat;
      if (n_alleles >= max_n_alleles) gabort("Too many alleles", n_alleles);
      if ((prev_n_alleles != -1) && (n_alleles != prev_n_alleles)) 
         gabort("No. alleles mismatch", n_alleles);
//      printf("n_alleles %d\n", n_alleles);       
//      for (i=0; i<4; i++) printf("%d ", ingroup[i]);
//      printf("\t");
      for (i=0; i<n_outgroup; i++)
      {
         stat = read_data_element(fptr, ingroup, outgroup, 1, i);
         if (stat==-99) gabort("Unexpected EOF", 0);
//         for (j=0; j<4; j++) printf("%d ", outgroup[j][i]);
      }
//      printf("\n");
      nsites++;
      config_ind = assign_observed_type(n_alleles, ingroup, outgroup, &n_config,
         n_outgroup, totsites, nsites);
      stat = skiptoendofline(fptr);
      if (stat == EOF) break;
      counter++;
      if (counter == 100000) counter = 0;
   }
   fclose(fptr);
   if (verbose)
   {
      printf("n_alleles %d sites %d\n", n_alleles, nsites);
      dump_configs(n_config, n_outgroup);
      for (i=0; i<=n_alleles; i++) printf("Alleles %d sites %d\n", i, totsites[i]);
   }
//   monitorinput();

   if (nrandom == 0) reps_with_random_start = 1; else reps_with_random_start = nrandom;
   for (i=1; i<=reps_with_random_start; i++)
   {
      if (nrandom != 0) set_up_random_starting_values(n_branch, k_est, model, &kappa,r6_parm);
//      for (j=0; j<n_branch; j++)               // Number of branches
//      {
//         printf("k%d %lf ", j, k_est[j]);
//      }
//      monitorinput();
      for (j = 1; j<=100; j++)
      {
         maxlogl = dolik_wrt_k_kappa_rate6_gsl(k_est, kappa, n_branch, r6_parm);
         random_start_ml[i-1] = maxlogl;
         if (verbose) printf("maxlogl %lf\n", maxlogl);
         if (j != 1)
         {
            diff_log_l = maxlogl - prev_log_l;
//          printf("diff_log_l %lf\n", diff_log_l); monitorinput();
            prev_log_l = maxlogl;
            if (diff_log_l<0.01)
            {
               if (verbose) printf("diff_log_l %lf prev_log_l %lf maxlogl %lf - breaking\n",
                  diff_log_l, prev_log_l, maxlogl);
//               monitorinput();
               random_start_ml[i-1] = maxlogl;
               break;
            }
         }
         else
         {
            prev_log_l = maxlogl;
         }
//         monitorinput();
      }
      if (verbose) printf("Random start %d maxlogl %lf\n", i, maxlogl);
      if ((current_ml == 0)||(maxlogl >= current_ml))
      {
         if (verbose) printf("current_ml %lf\n", current_ml);
         current_ml = maxlogl;
         store_ml_estimates(n_branch, k_est, k_est_saved, model, kappa, &kappa_saved,
            r6_parm, r6_parm_saved);
      }
      if (reps_with_random_start > 1)
      {
         printf("Run %3d ML %2.6lf ", i, maxlogl);
         for (j=0; j<n_branch; j++)               // Number of branches
         {
            printf("k%d %2.4lf ", j, k_est[j]);
         }
         if (model == kimura)
         {
            printf("kappa %2.4lf", kappa);
         }
         printf("\n");
         if (model == rate6)
         {
            printf("  ");
            for (j=0; j<6; j++)            // Number of parameters = 6
            {
               printf("r6[%d] %2.4lf ", j, r6_parm[j]);
            }
            printf("\n");
         }
      }
//      monitorinput();
   }
   printf("Overall ML %2.4lf ", maxlogl);
   for (i=0; i<n_branch; i++)               // Number of branches
   {
      k_est[i] = k_est_saved[i];
      printf("k%d %2.4lf ", i, k_est[i]);
   }
   if (model == kimura)
   {
      kappa = kappa_saved;
      printf("kappa %2.4lf", kappa);
   }
   printf("\n");
   if (model == rate6)
   {
      printf("  ");
      for (i=0; i<6; i++)            // Number of parameters = 6
      {
         r6_parm[i] = r6_parm_saved[i];
         printf("r6[%d] %2.4lf ", i, r6_parm[i]);
      }
      printf("\n");
   }
//   monitorinput();
   trace_flag = 0;
   i = n_alleles/2;
   start = i;
   if (odd(n_alleles))
   {
      start++;
   }
//   printf("start %d\n", start);
//   monitorinput();


   for (ind1=start; ind1<=n_alleles; ind1++)
//   for (ind1=n_alleles; ind1<=n_alleles; ind1++)       // TESTING
   {
      iter = 0;
      nevals = 1000;
      np = 1;
/* Initial vertex size vector */
      ss = gsl_vector_alloc (np);
/* Set all step sizes to stepsize */
      gsl_vector_set_all (ss, stepsize);
/* Starting point */
      x = gsl_vector_alloc (np);
      gsl_vector_set (x, 0, 0.5);      // uSFS element starting value
      ex4_fn.f = &compute_log_likelihood_wrt_sfs_p_gsl;
      ex4_fn.n = np;
      ex4_fn.params = (void *)&par;
      s = gsl_multimin_fminimizer_alloc (gsl_multimin_fminimizer_nmsimplex , np);
      gsl_multimin_fminimizer_set (s, &ex4_fn, x, ss);
      do
      {
         iter++;
         status = gsl_multimin_fminimizer_iterate(s);
         if (status)
         break;
         size = gsl_multimin_fminimizer_size (s);
         status = gsl_multimin_test_size (size, convcrit);
         if (status == GSL_SUCCESS)
         {
//          printf ("converged to minimum at\n");
         }
//        printf ("%5d ", iter);
         for (j = 0; j < np; j++)
         {
//            printf ("%10.3e ", gsl_vector_get (s->x, j));
         }
//       printf ("f() = %7.3f size = %.3f\n", s->fval, size);
//       monitorinput();
      }
      while (status == GSL_CONTINUE && iter < nevals);
      sfs_p_est = gsl_vector_get(s->x, 0);
//      printf("Simplex sfs_p_est %lf\n", sfs_p_est);
//      monitorinput();
//      res = golden(0, 0.5, 1.0, compute_log_likelihood_wrt_sfs_p, tol, &xmin, &neval);
//      sfs_p_est = xmin;
      sfs_p_est_vec[ind1] = sfs_p_est;
      sites_in_sfs_cat = totsites[ind1];
      sfs[i] = sfs_p_est*(double)sites_in_sfs_cat;
      if (i==ind1) sfs[ind1] += (1.0 - sfs_p_est)*(double)sites_in_sfs_cat;
      else sfs[ind1] = (1.0 - sfs_p_est)*(double)sites_in_sfs_cat;
//      printf("simplex maximizaton: sites_in_sfs_cat %d sfs_p_est %lf sfs[%d] %lf sfs[%d] %lf\n", sites_in_sfs_cat, sfs_p_est, i, sfs[i], ind1, sfs[ind1]);
//      monitorinput();
      i--;
      gsl_vector_free(x);
      gsl_vector_free(ss);
      gsl_multimin_fminimizer_free (s);
   }
   outfile_ptr = openforwritesilent(output_file, "w");
   for (i=0; i<=n_alleles; i++)
   {
//      printf("%f", sfs[i]);
//      if (i!=n_alleles) printf(",");
      fprintf(outfile_ptr, "%f", sfs[i]);
      if (i!=n_alleles) fprintf(outfile_ptr, ",");
   }
//   printf("\n");
   fprintf(outfile_ptr, "\n");
//   for (i=0; i<n_config; i++)
//   {
//      printf("i %lf\n", data[i].p_anc_major);
//   }

// Now recalculate the Ancestral likelihood probabilities including prior information
   i = n_alleles/2;
   start = i;
   if (odd(n_alleles))
   {
      start++;
   }
//   printf("start %d\n", start);
//   monitorinput();

   for (ind1=start; ind1<n_alleles; ind1++)
//   for (ind1=n_alleles; ind1<=n_alleles; ind1++)       // TESTING
   {
      for (j=0; j<2*n_outgroup - 1; j++)              // Number of branches
      {
         compute_poisson_vec(k_est[j], poiss_p, j);
      }
      if (model == rate6)
      {
         n_branch = 2*n_outgroup - 1;
         compute_r6_probs(n_branch, k_est, poiss_p_r6, r6_parm, r6_p);
      }
      n_tree = pow(4, n_outgroup - 1);
      if (n_tree >= max_n_tree)
      {
         printf("n_tree (%d) exceeds max_n_tree (%d). Terminating\n", n_tree, max_n_tree);
         gabort("Program aborting", 0);
      }
      static double ptree_dum[max_n_tree];
      sites_in_sfs_cat = totsites[ind1];
      rdum = compute_log_likelihood_wrt_sfs_p_seg(poiss_p, sfs_p_est_vec[ind1], n_tree,
         model, kappa, r6_parm, poiss_p_r6, r6_p, ptree_dum, 1);  // 0 for TESTING, 1 FOR LIVE
      i--;
   }

   fptr = openforreadsilent(data_file);
   n_config = 0;
   outfile_ptr = openforwritesilent(output_file_p_anc, "w");
   fprintf(outfile_ptr, "0 %s %2.2f\n", argv[0], version_number);
   fprintf(outfile_ptr, "0 sites %d\n", nsites);
   fprintf(outfile_ptr, "0 model %d\n", model);
   fprintf(outfile_ptr, "0 ML %lf\n", maxlogl);
   fprintf(outfile_ptr, "0 ML-random-starts ");
   for (i=0; i<reps_with_random_start; i++)
   {
      fprintf(outfile_ptr, "%lf ", random_start_ml[i]);
   }
   fprintf(outfile_ptr, "\n");
   fprintf(outfile_ptr, "0 Rates: ");
   for (i=0; i<n_branch; i++)               // Number of branches
   {
      fprintf(outfile_ptr,  "k%d %lf ", i, k_est[i]);
   }
   fprintf(outfile_ptr, "\n");
   if (model == kimura)
   {
      fprintf(outfile_ptr, "0 kappa %lf\n", kappa);
   }
   else if (model == rate6)
   {
      for (i=0; i<6; i++)            // Number of parameters = 6
      {
         fprintf(outfile_ptr, "0 r6[%d] %lf ", i, r6_parm[i]);
      }
      fprintf(outfile_ptr, "\n");
   }
   fprintf(outfile_ptr, "0 Site Code P-major-ancestral P-trees[A,C,G,T]\n");

   n_tree = pow(4, n_outgroup -1);
   sind = 0;
   for (;;)
   {
      stat = read_data_element(fptr, ingroup, outgroup, alleles_to_output, -1);
      if (stat==-99) break;
      for (i=0; i<n_outgroup; i++)
      {
         stat = read_data_element(fptr, ingroup, outgroup, 1, i);
         if (stat==-99) gabort("Unexpected EOF", 0);
//         for (j=0; j<4; j++) printf("%d ", outgroup[j][i]);
      }
//      printf("\n");
      config_ind = assign_observed_type(n_alleles, ingroup, outgroup, &n_config, n_outgroup, totsites, sind);
      p = data[config_ind].p_anc_major;
//      printf("config_ind %d p %lf\n", config_ind, p);
      sind++;
      fprintf(outfile_ptr, "%d %d %lf ", sind, config_ind, p);
      ptot = 0.0;
      for (i=1; i<=n_tree; i++)
      {
         ptot += data[config_ind].ptree[i];
      }
      for (i=1; i<=n_tree; i++)
      {
         fprintf(outfile_ptr, "%lf ", data[config_ind].ptree[i]/ptot);
      }
      fprintf(outfile_ptr, "\n");
      stat = skiptoendofline(fptr);
      if (stat == EOF) break;
   }
   fclose(fptr);
   fclose(outfile_ptr);
   writeseed(seedfile);
   exit(0);
}
