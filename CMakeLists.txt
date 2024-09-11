cmake_minimum_required(VERSION 2.8.3)
project(butia_quiz)

find_package(catkin REQUIRED COMPONENTS
  rospy 
  std_msgs 
  message_generation
)

catkin_python_setup()

## Generate services in the 'srv' folder
add_service_files(
  FILES
    ButiaQuizComm.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  nodes/butia_quiz_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
