/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // pf.init(sense_x, sense_y, sense_theta, sigma_pos)
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  if (!is_initialized){
    num_particles = 42;  // relatively small number because we have an initial GPS geuss of position

    std::normal_distribution<double> ndist_x(x, std[0]);
    std::normal_distribution<double> ndist_y(y, std[1]);
    std::normal_distribution<double> ndist_theta(theta, std[2]);

    // init particles around pos with noise
    for(int i = 0; i < num_particles; i++){
      // new PArticle instance
      Particle p;
      p.id = i;
      p.x = ndist_x(generator);
      p.y = ndist_y(generator);
      p.theta = ndist_theta(generator);
      p.weight = 1.0;

      // add particle to Particles
      particles.push_back(p);
    }

    // set initialized
    is_initialized = true;
    std::cout << "init particle filter" << std::endl;

}

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  // create error dist outside of loop, they stay the same
  std::normal_distribution<double> ndist_x(0, std_pos[0]);
  std::normal_distribution<double> ndist_y(0, std_pos[1]);
  std::normal_distribution<double> ndist_theta(0, std_pos[2]);

  // predict
  for (int i=0; i<num_particles; i++) {

  	double theta = particles[i].theta;
    if ( fabs(yaw_rate) < epsilon ) { // avoid dividing by 0
      particles[i].x += velocity * delta_t * cos( theta );
      particles[i].y += velocity * delta_t * sin( theta );
    } else {
      particles[i].x += velocity / yaw_rate * ( sin( theta + yaw_rate * delta_t ) - sin( theta ) );
      particles[i].y += velocity / yaw_rate * ( cos( theta ) - cos( theta + yaw_rate * delta_t ) );
      particles[i].theta += yaw_rate * delta_t;
    }
    // accounting for measurement noise
    particles[i].x += ndist_x(generator);
    particles[i].y += ndist_y(generator);
    particles[i].theta += ndist_theta(generator);

  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  // Take observation and loop over predictions to find landmark with smallest distance
  for (unsigned int i=0; i<observations.size(); i++){
    double min_dist = sensor_range * 2.0; // ralistic distance should not be larger, given that startng position is constrained

    LandmarkObs obs = observations[i];

    // init id of landmark id
    int lm_id = -1;
    
    for (unsigned int j=0; j<predicted.size(); j++) {
      LandmarkObs pred = predicted[j];
      
      // dist b/w observed and pred landmark
      double distance = dist(obs.x, obs.y, pred.x, pred.y);

      // only update if distance is smaller
      if (distance < min_dist) {
        min_dist = distance;
        lm_id = pred.id;
      }
    }

    // set the observation's id to the nearest predicted landmark's id
    observations[i].id = lm_id;

  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (int i=0; i < num_particles; i++) {
    double theta = particles[i].theta;
    // get all landmarks within sensor range
    vector<LandmarkObs> visibleLandmarks;
    for(unsigned int j=0; j < map_landmarks.landmark_list.size(); j++) {
      double lm_x = map_landmarks.landmark_list[j].x_f;
      double lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;
      if (pow(particles[i].x - lm_x, 2) + pow(particles[i].y - lm_y, 2) <= pow(sensor_range, 2) ) {
        visibleLandmarks.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
      }
    }

    // Transform obs coords to map
    vector<LandmarkObs> map_obs;
    for(unsigned int j = 0; j < observations.size(); j++) {
      map_obs.push_back(LandmarkObs{ observations[j].id, 
                    cos(theta)*observations[j].x - sin(theta)*observations[j].y + particles[i].x,
                    sin(theta)*observations[j].x + cos(theta)*observations[j].y + particles[i].y,
                    });
    }

    // get most likely observation
    dataAssociation(visibleLandmarks, map_obs);

    // recalc weights based on associated landmarks
    particles[i].weight = 1;
    for(unsigned int j = 0; j < map_obs.size(); j++) {
      int lm_id = map_obs[j].id;
      double lm_x, lm_y;
      unsigned int nl = 0;
      bool match = false;
      while( !match && nl < visibleLandmarks.size() ) {
        if ( visibleLandmarks[nl].id == lm_id) {
          match = true;
          lm_x = visibleLandmarks[nl].x;
          lm_y = visibleLandmarks[nl].y;
        }
        nl++;
      }

      // Calculating weight.
      double weight = ( 1/(2*M_PI*std_landmark[0]*std_landmark[1])) * 
            exp( -( pow(map_obs[j].x - lm_x, 2)/(2 * pow(std_landmark[0],2) ) + 
            (pow(map_obs[j].y - lm_y, 2) / (2*std_landmark[1]*std_landmark[1])) ) );
      if (!weight) {      // in case of particles with 0 weight
        particles[i].weight *= epsilon;
      } else {
        particles[i].weight *= weight;
      }
    }
  }

}

void ParticleFilter::resample() {
  // get all weights
  std::vector<double> ww;
  for (auto const &w : particles){
    ww.push_back(w.weight);
  }
  // generate weighted deistribution
  std::discrete_distribution<unsigned int> wdist(std::begin(ww), std::end(ww));
  //resample based on wdist
  std::vector<Particle> tempParticles;
  Particle tp;
  for (int i=0; i<num_particles; i++){
    tp = particles[wdist(generator)];
    tp.id = i;
    tempParticles.push_back(tp);
  }
  particles = tempParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}