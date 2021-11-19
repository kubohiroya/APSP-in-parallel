package jp.ac.cuc.hiroya.apsp;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.TimeZone;

public class DateTimeUtil {
    static TimeZone timeZoneJP = TimeZone.getTimeZone("Asia/Tokyo");
    static SimpleDateFormat sdf = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");

    static {
        sdf.setTimeZone(timeZoneJP);
    }

    static String getYYMMDDHHMM() {
        return sdf.format(new Date());
    }
}
