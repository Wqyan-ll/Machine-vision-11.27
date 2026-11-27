#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

cv::Mat readImage(const std::string& path)
{
    cv::Mat image = cv::imread(path);
    if (image.empty())
        throw std::runtime_error("Cannot read image: " + path);
    return image;
}

std::vector<cv::Point> largestContour(const cv::Mat& mask)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty())
        throw std::runtime_error("No object boundary found.");
    return *std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
}

// Order the corners as top-left, top-right, bottom-right, bottom-left.
std::vector<cv::Point2f> findCorners(const cv::Mat& mask)
{
    const auto contour = largestContour(mask);
    std::vector<cv::Point> hull, polygon;
    cv::convexHull(contour, hull);
    cv::approxPolyDP(hull, polygon, 0.02 * cv::arcLength(hull, true), true);
    if (polygon.size() != 4 || !cv::isContourConvex(polygon))
        throw std::runtime_error("Expected four corners. Check the foreground mask.");

    std::vector<cv::Point2f> corners(4);
    auto sum = [](const cv::Point& p) { return p.x + p.y; };
    auto difference = [](const cv::Point& p) { return p.x - p.y; };
    corners[0] = *std::min_element(polygon.begin(), polygon.end(),
        [&](const cv::Point& a, const cv::Point& b) { return sum(a) < sum(b); });
    corners[2] = *std::max_element(polygon.begin(), polygon.end(),
        [&](const cv::Point& a, const cv::Point& b) { return sum(a) < sum(b); });
    corners[1] = *std::max_element(polygon.begin(), polygon.end(),
        [&](const cv::Point& a, const cv::Point& b) { return difference(a) < difference(b); });
    corners[3] = *std::min_element(polygon.begin(), polygon.end(),
        [&](const cv::Point& a, const cv::Point& b) { return difference(a) < difference(b); });
    return corners;
}

cv::Mat rectify(const cv::Mat& image, const std::vector<cv::Point2f>& corners)
{
    const int width = cvRound(std::max(cv::norm(corners[1] - corners[0]),
        cv::norm(corners[2] - corners[3])));
    const int height = cvRound(std::max(cv::norm(corners[3] - corners[0]),
        cv::norm(corners[2] - corners[1])));
    if (width < 2 || height < 2)
        throw std::runtime_error("Invalid rectification size.");
    const std::vector<cv::Point2f> destination = {
        {0, 0}, {static_cast<float>(width - 1), 0},
        {static_cast<float>(width - 1), static_cast<float>(height - 1)},
        {0, static_cast<float>(height - 1)}
    };
    cv::Mat result;
    cv::warpPerspective(image, result,
        cv::getPerspectiveTransform(corners, destination), cv::Size(width, height),
        cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    return result;
}

cv::Mat markCorners(const cv::Mat& image, const std::vector<cv::Point2f>& corners)
{
    cv::Mat marked = image.clone();
    for (int i = 0; i < 4; ++i)
    {
        cv::line(marked, corners[i], corners[(i + 1) % 4], cv::Scalar(0, 255, 255), 2);
        cv::circle(marked, corners[i], 5, cv::Scalar(0, 0, 255), cv::FILLED);
    }
    return marked;
}

cv::Mat panel(const std::vector<cv::Mat>& images, const std::vector<std::string>& titles,
    int cellWidth, int cellHeight)
{
    cv::Mat output(cellHeight + 38, cellWidth * static_cast<int>(images.size()),
        CV_8UC3, cv::Scalar(32, 32, 32));
    for (size_t i = 0; i < images.size(); ++i)
    {
        cv::Mat color;
        if (images[i].channels() == 1)
            cv::cvtColor(images[i], color, cv::COLOR_GRAY2BGR);
        else
            color = images[i];
        const double scale = std::min((cellWidth - 16.0) / color.cols,
            (cellHeight - 16.0) / color.rows);
        cv::Mat resized;
        cv::resize(color, resized, cv::Size(), scale, scale, cv::INTER_AREA);
        const int x = static_cast<int>(i) * cellWidth + (cellWidth - resized.cols) / 2;
        const int y = 38 + (cellHeight - resized.rows) / 2;
        resized.copyTo(output(cv::Rect(x, y, resized.cols, resized.rows)));
        cv::putText(output, titles[i], cv::Point(static_cast<int>(i) * cellWidth + 12, 25),
            cv::FONT_HERSHEY_SIMPLEX, 0.57, cv::Scalar::all(245), 1, cv::LINE_AA);
    }
    return output;
}

void save(const std::string& name, const cv::Mat& image)
{
    if (!cv::imwrite("results/" + name + ".png", image))
        throw std::runtime_error("Cannot save " + name);
}

int main(int argc, char* argv[])
{
    const bool showWindows = !(argc == 2 && std::string(argv[1]) == "--no-gui");
    try
    {
        // Paths are relative to the project directory, as configured in Visual Studio.
        cv::Mat part = readImage("lQDPJxGjfTT5XM0wbLDjns2yNhd3kwrYkdG5QcYA_108_48.jpg");
        const cv::Mat portrait = readImage(
            "lQDPJxjtxbANnM3NAnTNAnSwI7dE3FlskucK2JHSO4kJAA_628_628.jpg");
        const cv::Mat book = readImage(
            "lQDPKHn_WxPKvc3NBQDNA8CwsS8VL5UyGc0K2JHS4nP5AA_960_1280.jpg");
        // Preserve the original pixels above. Upsampling only makes detection and display easier.
        cv::resize(part, part, cv::Size(), 4, 4, cv::INTER_CUBIC);
        cv::Mat gray, edges, mask;
        cv::cvtColor(part, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0.8);
        cv::Canny(gray, edges, 40, 100);
        cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        const auto box = cv::minAreaRect(largestContour(mask));
        cv::Point2f vertices[4];
        box.points(vertices);
        int longest = 0;
        for (int i = 1; i < 4; ++i)
            if (cv::norm(vertices[(i + 1) % 4] - vertices[i])
                > cv::norm(vertices[(longest + 1) % 4] - vertices[longest]))
                longest = i;
        const cv::Point2f direction = vertices[(longest + 1) % 4] - vertices[longest];
        const double expectedNormal = std::atan2(direction.y, direction.x) + CV_PI / 2;

        // Standard Hough transform. Choose the two lines nearest the long outer sides.
        std::vector<cv::Vec2f> lines;
        cv::HoughLines(edges, lines, 1, CV_PI / 360, 70);
        cv::Mat fullLines = part.clone();
        for (int side : {longest, (longest + 2) % 4})
        {
            const cv::Point2f midpoint = (vertices[side] + vertices[(side + 1) % 4]) * 0.5f;
            double bestDistance = 1e9;
            cv::Vec2f bestLine;
            for (const auto& line : lines)
            {
                if (std::abs(std::cos(line[1] - expectedNormal)) < std::cos(5 * CV_PI / 180))
                    continue;
                const double distance = std::abs(midpoint.x * std::cos(line[1])
                    + midpoint.y * std::sin(line[1]) - line[0]);
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestLine = line;
                }
            }
            if (bestDistance > 12)
                throw std::runtime_error("Could not detect both long edges of the part.");
            const double a = std::cos(bestLine[1]), b = std::sin(bestLine[1]);
            const cv::Point2d origin(a * bestLine[0], b * bestLine[0]);
            cv::line(fullLines, origin + cv::Point2d(-1000 * b, 1000 * a),
                origin - cv::Point2d(-1000 * b, 1000 * a), cv::Scalar(0, 255, 255), 2);
        }

        // Probabilistic Hough transform returns finite segments, including the inner slot.
        std::vector<cv::Vec4i> segments;
        cv::HoughLinesP(edges, segments, 1, CV_PI / 360, 22, 48, 10);
        cv::Mat segmentImage = part.clone();
        for (const auto& line : segments)
            cv::line(segmentImage, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]),
                cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        // Saturation separates the colored portrait from its white triangular margins.
        cv::Mat hsv, portraitMask, bookMask;
        cv::cvtColor(portrait, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0, 40, 25), cv::Scalar(179, 255, 255), portraitMask);
        const auto portraitCorners = findCorners(portraitMask);
        const cv::Mat portraitResult = rectify(portrait, portraitCorners);

        // Isolate the blue cover, then fill gaps through its convex outer boundary.
        cv::cvtColor(book, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(85, 45, 35), cv::Scalar(125, 255, 255), bookMask);
        cv::morphologyEx(bookMask, bookMask, cv::MORPH_CLOSE,
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
        const auto bookCorners = findCorners(bookMask);
        const cv::Mat bookResult = rectify(book, bookCorners);

        const cv::Mat partPanel = panel({part, edges, fullLines, segmentImage},
            {"Original (4x)", "Canny", "Hough: two long edges", "HoughP: line segments"}, 350, 210);
        const cv::Mat portraitPanel = panel({portrait, markCorners(portrait, portraitCorners), portraitResult},
            {"Portrait original", "Detected corners", "Portrait corrected"}, 466, 430);
        const cv::Mat bookPanel = panel({book, markCorners(book, bookCorners), bookResult},
            {"Book original", "Detected corners", "Book corrected"}, 466, 510);
        cv::utils::fs::createDirectories("results");
        save("01_canny", edges);
        save("02_hough_lines", fullLines);
        save("03_hough_segments", segmentImage);
        save("04_portrait_corrected", portraitResult);
        save("05_book_corrected", bookResult);
        save("part_comparison", partPanel);
        save("portrait_comparison", portraitPanel);
        save("book_comparison", bookPanel);
        std::cout << "Hough candidates: " << lines.size() << ", segments: " << segments.size()
            << "\nPortrait: " << portraitResult.cols << " x " << portraitResult.rows
            << "\nBook: " << bookResult.cols << " x " << bookResult.rows
            << "\nSaved results to ./results/\n";
        if (showWindows)
        {
            for (const auto& item : std::vector<std::pair<std::string, cv::Mat>>{
                {"Part - Canny and Hough", partPanel}, {"Portrait correction", portraitPanel},
                {"Book correction", bookPanel}})
            {
                cv::namedWindow(item.first, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
                cv::imshow(item.first, item.second);
                cv::resizeWindow(item.first, 1200, cvRound(item.second.rows * 1200.0 / item.second.cols));
            }
            cv::waitKey(0);
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << "\n";
        return 1;
    }
}

