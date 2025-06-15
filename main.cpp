#include <glog/logging.h>
#include <gtest/gtest.h>

#include <iostream>
int main(int argc, char *argv[]) {
	testing::InitGoogleTest(&argc, argv);
	google::InitGoogleLogging("jinfer");
	FLAGS_log_dir = "log";
	FLAGS_alsologtostderr = true;

	LOG(INFO) << "Start test...\n";
	return RUN_ALL_TESTS();
}