#!/bin/bash

set -ex

# set paths
# no user interaction needed here
registry_templates_path="templates/registry"
endpoint_templates_path="templates/endpoint"
parameters='@parameters.json'
template="template.json"
# docker paths
# meant to be built from root of the repo
dockerfile_path="azure/docker/dockerfile"
docker_path=".."
starting=$(pwd)

# declare all inputs in this script
# Azure inputs
resource_group="CerberusTestingDE"
# API and docker inputs
docker_image_name="docclasssfltesting"
registry_name=$docker_image_name"Registry"
# CIDR whitelist
cidr_whitelist="73.219.252.245/32"

# deploy registry

cd $registry_templates_path

parameters_file_name=$(echo $parameters | cut -c 2-)

# set parameter for name of the registry from name of the docker image
variable=$registry_name
jq --arg variable "$variable" '.parameters.name.value = $variable' $parameters_file_name >tmpparms && cat tmpparms >$parameters_file_name

# get deployment name from
deployment_name=$(cat $parameters_file_name | jq '.parameters.name.value' | cut -c 2- | rev | cut -c 2- | rev)

echo "Deploying: "$deployment_name

az deployment group create \
     --name $deployment_name \
     --resource-group $resource_group \
     --template-file $template \
     --parameters $parameters

cd $starting

echo "Waiting 60s to allow registry to come online"
sleep 60

# get registry login
registry=$(echo $deployment_name | tr '[:upper:]' '[:lower:]')
registry_username=$deployment_name # $(az acr credential show -n $deployment_name --query username | cut -c 2- | rev | cut -c 2- | rev )
registry_username_proper=$(az acr credential show -n $deployment_name --query username | cut -c 2- | rev | cut -c 2- | rev)
registry_password=$(az acr credential show -n $deployment_name --query passwords[0].value | cut -c 2- | rev | cut -c 2- | rev)
az acr login --name $registry --user $registry_username --password $registry_password

# build docker
cd $docker_path

docker_tag=$registry".azurecr.io/"$docker_image_name
docker build -t $docker_tag -f $dockerfile_path .

# push docker
docker push $docker_tag

# go to start directory
cd $starting

# deploy endpoint
cd $endpoint_templates_path

parameters_file_name=$(echo $parameters | cut -c 2-)

# update the parameters file for endpoint with the values for the docker image
registry_url="https://"$registry".azurecr.io"

variable=$registry_username
jq --arg variable "$variable" '.parameters.dockerRegistryUsername.value = $variable' $parameters_file_name >tmpparms && cat tmpparms >$parameters_file_name
variable=$registry_password
jq --arg variable "$variable" '.parameters.dockerRegistryPassword.value = $variable' $parameters_file_name >tmpparms && cat tmpparms >$parameters_file_name
variable=$registry_url
jq --arg variable "$variable" '.parameters.dockerRegistryUrl.value = $variable' $parameters_file_name >tmpparms && cat tmpparms >$parameters_file_name
# linuxfx version
linuxFxVersion="DOCKER|"$docker_tag":latest"
variable=$linuxFxVersion
jq --arg variable "$variable" '.parameters.linuxFxVersion.value = $variable' $parameters_file_name >tmpparms && cat tmpparms >$parameters_file_name

# hosting plan name
hosting_plan_name=$deployment_name"hostingplan"
variable=$hosting_plan_name
jq --arg variable "$variable" '.parameters.hostingPlanName.value = $variable' $parameters_file_name >tmpparms && cat tmpparms >$parameters_file_name

# deployment name
api_name=$docker_image_name #"-"$(uuid)
variable=$api_name
jq --arg variable "$variable" '.parameters.name.value = $variable' $parameters_file_name >tmpparms && cat tmpparms >$parameters_file_name

deployment_name=$(cat $parameters_file_name | jq '.parameters.name.value' | cut -c 2- | rev | cut -c 2- | rev)

# CIDR whitelist
variable=$cidr_whitelist
jq --arg variable "$variable" '.variables.allowedCIDR = $variable' $template >tmpplate # && cat tmpplate > $template

echo "Deploying: "$deployment_name

az deployment group create \
     --name $deployment_name \
     --resource-group $resource_group \
     --template-file $template \
     --parameters $parameters

# restart api app in case this is deploying over existing app
az webapp restart --name $api_name --resource-group $resource_group
# wait for it to reset
sleep 60
# allow non-zero exit codes
set +e
# repeat loop until curl|grep returns exit code 0
until curl -m 5 https://$api_name.azurewebsites.net/ | grep "memory_usage_percent"; do
     echo "waiting for API to become ready"
     sleep 30
done

echo "API is ready"
