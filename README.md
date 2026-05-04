## Web App

**Live Pitch Recommendation: [andersen-pickard.shinyapps.io/pitch_app/](https://andersen-pickard.shinyapps.io/pitch_app/)**

## Author
**Andersen Pickard, Rice University**
_[Click here](https://andersenpickard.wixsite.com/home) for online portfolio_

## Abstract

Pitch selection in Major League Baseball has remained a largely intuitive process, historically delegated to the catcher despite baseball’s broader embrace of data analytics. This paper proposes and implements a machine learning framework to quantify the cost of suboptimal pitch selection decisions and evaluate the performance of pitchers, catchers, teams, and pitcher-catcher batteries. Using Statcast pitch-level data from the 2021 through 2025 MLB seasons, a LightGBM gradient boosting model is trained to predict runs lost as a function of game situation and pitch characteristics. The trained model is then applied to generate counterfactual pitch scenarios for every pitch thrown in 2025, estimating the expected run value of each pitch type a pitcher could have thrown in a given situation. The pitch type with the lowest predicted run value is identified as the optimal selection, and the difference between the predicted run value of the actual pitch and the optimal pitch is quantified as runs lost. Aggregated across pitchers, catchers, teams, and batteries, these pitch-level estimates reveal substantial variation in pitch selection quality across the league. League-wide, MLB players selected the optimal pitch type just 25.2% of the time in 2025. The findings suggest that predictive modeling tools capable of transmitting optimal pitch calls from the dugout to the field in real time could offer organizations a meaningful and largely untapped competitive advantage over their opponents, resulting in potentially 100+ additional runs per season.
