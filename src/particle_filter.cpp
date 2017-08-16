/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 20;
  
  default_random_engine gen;
  // Create a normal (Gaussian) distribution for x, y, and psi
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  for(int i = 0; i < num_particles; i++) {
    Particle new_particle = Particle();
    new_particle.id = i;
    new_particle.x = dist_x(gen);
    new_particle.y = dist_y(gen);
    new_particle.theta = dist_theta(gen);
    new_particle.weight = 1;
    // initialize weights and particles vectors
    weights.push_back(1);
    particles.push_back(new_particle);
    cout << "new particle " << i << "  = [" << new_particle.x << ", " << new_particle.y << ", " << new_particle.theta << "]" << endl;
  }
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
  
  default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_psi(0, std_pos[2]);
  
  for(int i = 0; i < num_particles; i++) {
    double old_theta = particles[i].theta;

    cout << "particle " << i << " before predict [" << particles[i].x << ", " << particles[i].y << ", " << particles[i].theta << "]" << endl;
    
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(old_theta) + dist_x(gen);
      particles[i].y += velocity * delta_t * sin(old_theta) + dist_y(gen);
      particles[i].theta += dist_psi(gen);
    } else {
      double new_theta = old_theta + delta_t * yaw_rate;
      particles[i].x += velocity/yaw_rate * (sin(new_theta) - sin(old_theta)) + dist_x(gen);
      particles[i].y += velocity/yaw_rate * (-cos(new_theta) + cos(old_theta)) + dist_y(gen);
      particles[i].theta = new_theta + dist_psi(gen);
    }
    
    cout << "particle " << i << " after predict [" << particles[i].x << ", " << particles[i].y << ", " << particles[i].theta << "]" << endl;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  
  double contrib_const = 1.0 / (2. * M_PI * std_landmark[0] * std_landmark[1]);
  
  for (int i = 0; i < particles.size(); i++) {
    std::vector<LandmarkObs> observations_transf;
    
    vector<double> distances;
    vector<double> contributions;
    
    double th = particles[i].theta;
    double cos_th = cos(th);
    double sin_th = sin(th);

    double x0 = particles[i].x;
    double y0 = particles[i].y;
    
    for (LandmarkObs obsrv : observations) {
      LandmarkObs obs_t;

      // Calculate map coordinates of observation
      obs_t.x = x0 + cos_th * obsrv.x - sin_th * obsrv.y;
      obs_t.y = y0 + sin_th * obsrv.x + cos_th * obsrv.y;
      observations_transf.push_back(obs_t);
      cout << "obsrv [" << obsrv.x << ", " << obsrv.y << "] transf [" << obs_t.x << ", " << obs_t.y << "]" << endl;
    
      // Calculate distance from observation (in map coordinates) to all landmarks
      vector <double> distances;
      for (Map::single_landmark_s landm : map_landmarks.landmark_list) {
        double distance = dist(landm.x_f, landm.y_f, obs_t.x, obs_t.y);
        distances.push_back(distance);
      }
      
      // identify the landmark closest to the observation - this is the "associated" landmark
      vector<double>::iterator result = min_element(begin(distances), end(distances));
      int min_index = distance(begin(distances), result);
      Map::single_landmark_s lm = map_landmarks.landmark_list[min_index];
      obs_t.id = lm.id_i; // this is not used in the logic

      // calculate observation weight based on coordinates of the associated landmark
      cout << "p " << i << " association coords [" << lm.x_f << ", " << lm.y_f << "] index = " << min_index << endl;
      double contribution = exp(-1.0 * ((obs_t.x - lm.x_f) * (obs_t.x - lm.x_f) / 2.0 / std_landmark[0] / std_landmark[0] +
                                        (obs_t.y - lm.y_f) * (obs_t.y - lm.y_f) / 2.0 / std_landmark[1] / std_landmark[1]));
      
      cout << "p " << i << " contribution " << contribution << endl;
      contributions.push_back(contrib_const * contribution);
    }
    
    // multiply all the contributions of each of the observations to achieve the multivariate gaussian probability
    double weight = 1;
    for (double contribution : contributions) weight *= contribution;
    cout << "weight for particle " << i << " = " << weight << " -------" << endl;
    particles[i].weight = weight;
    weights[i] = weight;
  }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  discrete_distribution<int> dist(weights.begin(), weights.end());
  vector<Particle> new_particles = {};
  for(int i = 0; i < num_particles; i++) {
    new_particles.push_back(particles[dist(gen)]);
  }
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
