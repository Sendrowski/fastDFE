#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

int inputcounter=0;
int random_number_generator_initialised_flag = 0;
gsl_rng * gsl_rng_r;
unsigned long int seed;

void monitorinput()
{
   int stat;
   inputcounter--;
   if (inputcounter <= 0)
   {
      printf("Enter int to continue ");
      stat = scanf("%d", &inputcounter);
   }
}

void gabort(char *s, int r)
{
   printf("ERROR %s %d\n", s, r);
   exit(1);
}


void initialise_uniform_generator()
{
/* create a generator chosen by the environment variable GSL_RNG_TYPE */
   const gsl_rng_type * T;
//   printf("initialise_uniform_generator\n");
   gsl_rng_env_setup();
   T = gsl_rng_default;
   gsl_rng_r = gsl_rng_alloc (T);
   random_number_generator_initialised_flag = 1;
}


void getseedquick(char *seedfile)
{
   FILE *seedfileptr;
   int stat;
   static int c = 1;
   if (c==1)         // first time routine called
   {
      initialise_uniform_generator();
      c = 0;
   }
   seedfileptr = fopen(seedfile, "r");
   if (seedfileptr==0)
   {
      printf("No seedfile, enter seed please ");
      stat = scanf("%lu", &seed);
   }
   else
   {
      stat = fscanf(seedfileptr, "%lu", &seed);
      fclose(seedfileptr);
   }
//   printf("getseedquick: Seed read %lu\n", seed);
   gsl_rng_set(gsl_rng_r, seed);
}

void terminate_uniform_generator()
{
   gsl_rng_free(gsl_rng_r);
}

double uniform()
{
   double res;
   if (!random_number_generator_initialised_flag)
      gabort("uniform() called, but !random_number_generator_initialised_flag", 0);
   res = gsl_ran_flat(gsl_rng_r,0.0,1.0);
   return(res);
}

FILE *openforreadsilent(char *str)
{
   FILE *f;
   f = fopen(str, "r");
   if (f==0)
   {
      printf("ERROR: File %s not found.\n", str);
      exit(1);
   }
   return(f);
}


void writeseed(char *seedfile)
{
   FILE *seedfileptr;
   double temp, x;
   x = uniform();
   temp = floor(x*100000000);
   seed = (unsigned long int)temp;
//   printf("Write seed %lu\n", seed);
   seedfileptr = fopen(seedfile, "w");
   fprintf(seedfileptr, "%lu\n", seed);
   fclose(seedfileptr);
   terminate_uniform_generator();
}

FILE *openforwritesilent(char *str, char *mode)
{
   FILE *f;
   f = fopen(str, mode);
   if (f==0)
   {
      printf("ERROR: Unable to open file %s for write. Mode = %s.\n", str, mode);
      exit(0);
   }
   if (mode[0]=='a')
   {
   }
   else if (mode[0]=='w')
   {
   }
   else gabort("Invalid mode given in openforwrite\n", 0);
   return(f);
}

int skiptoendofline(FILE *fptr)
{
   int status;
   char ch;
   do
   {
      status = fscanf(fptr, "%c", &ch);
      if (status==EOF) break;
/*      printf("skiptoendofline ch %c\n", ch);*/
   }
   while (ch!='\n');
   return(status);
}

int odd(int i)
{
   int x;
   x = i/2;
   if ((double)(i)/2.0 - (double)x == 0.0) return(0); else return(1);
}
