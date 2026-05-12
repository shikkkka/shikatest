import requests
import json
from typing import Dict, Optional


# ================== IP数据云数据模型 ==================
class Street:
    def __init__(self, obj: Dict):
        self.lng = obj.get("lng")
        self.lat = obj.get("lat")
        self.province = obj.get("province")
        self.city = obj.get("city")
        self.district = obj.get("district")
        self.street = obj.get("street")
        self.radius = obj.get("radius")
        self.zip_code = obj.get("zip_code")

    def log(self):
        print(f"经度: {self.lng} 纬度: {self.lat}")
        print(f"省份: {self.province} 城市: {self.city} 区县: {self.district}")
        print(f"街道: {self.street} 邮编: {self.zip_code} 半径: {self.radius}")


class Location:
    def __init__(self, obj: Dict):
        self.area_code = obj.get("area_code")
        self.city = obj.get("city")
        self.city_code = obj.get("city_code")
        self.continent = obj.get("continent")
        self.country = obj.get("country")
        self.country_code = obj.get("country_code")
        self.district = obj.get("district")
        self.elevation = obj.get("elevation")
        self.ip = obj.get("ip")
        self.isp = obj.get("isp")
        self.latitude = obj.get("latitude")
        self.longitude = obj.get("longitude")
        self.multi_street = [Street(s) for s in obj.get("multi_street", [])]
        self.province = obj.get("province")
        self.street = obj.get("street")
        self.time_zone = obj.get("time_zone")
        self.weather_station = obj.get("weather_station")
        self.zip_code = obj.get("zip_code")

    def log(self):
        print(f"IP: {self.ip}")
        print(f"地理位置: {self.continent}/{self.country}({self.country_code})")
        print(f"行政区划: {self.province} {self.city} {self.district}")
        print(f"经纬度: {self.longitude}, {self.latitude}")
        print(f"ISP: {self.isp} 时区: {self.time_zone}")
        print("历史街道记录:")
        for street in self.multi_street:
            street.log()


# ================== 配置项 ==================
IPDATA_CLOUD_API_KEY = "76f551e71c2b11f0b3fd00163e167ffb"


# ================== IP数据云服务 ==================
def query_ipdatacloud(ip: str, query_type: int = 0) -> Optional[Location]:
    """查询IP数据云信息
    Args:
        ip: 目标IP
        query_type: 0-基础定位 1-区县 2-风险 3-应用场景
    """
    base_url = "https://api.ipdatacloud.com/v2/query"
    params = {
        "ip": ip,
        "key": IPDATA_CLOUD_API_KEY,
        "lang": "CN"
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()

        if data.get("code") != 200:
            print(f"API错误: {data.get('msg')}")
            return None

        # 根据类型解析数据
        if query_type <= 1:
            location_data = data["data"].get("location", {})
            return Location(location_data)
        # 其他类型处理（需补充Risk/Scenes类）
        # elif query_type == 2:
        #     risk_data = data["data"].get("risk", {})
        #     return Risk(risk_data)
        # else:
        #     scenes_data = data["data"].get("scenes", {})
        #     return Scenes(scenes_data)

    except Exception as e:
        print(f"请求异常: {str(e)}")
        return None


# ================== 使用示例 ==================
if __name__ == "__main__":
    test_ips = ["8.8.8.8", "114.114.114.114"]

    for ip in test_ips:
        print(f"\n===== 正在查询 {ip} =====")
        location = query_ipdatacloud(ip)
        if location:
            location.log()
        else:
            print("查询失败")